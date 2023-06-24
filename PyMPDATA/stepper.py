""" MPDATA iteration logic """
import sys
import warnings
from functools import lru_cache

import numba
import numpy as np
from numba.core.errors import NumbaExperimentalFeatureWarning

from .impl.clock import clock
from .impl.enumerations import ARG_DATA, IMPL_BC, IMPL_META_AND_DATA
from .impl.formulae_antidiff import make_antidiff
from .impl.formulae_axpy import make_axpy
from .impl.formulae_flux import make_flux_first_pass, make_flux_subsequent
from .impl.formulae_laplacian import make_laplacian
from .impl.formulae_nonoscillatory import make_beta, make_correction, make_psi_extrema
from .impl.formulae_upwind import make_upwind
from .impl.meta import _Impl
from .impl.traversals import Traversals
from .options import Options


class Stepper:
    """MPDATA stepper specialised for given options, dimensionality and optionally grid
    (instances of Stepper can be shared among `Solver`s)"""

    def __init__(
        self,
        *,
        options: Options,
        n_dims: (int, None) = None,
        non_unit_g_factor: bool = False,
        grid: (tuple, None) = None,
        n_threads: (int, None) = None,
        left_first: bool = True,
        buffer_size: int = 0
    ):
        if n_dims is not None and grid is not None:
            raise ValueError()
        if n_dims is None and grid is None:
            raise ValueError()
        if grid is None:
            grid = tuple([-1] * n_dims)
        if n_dims is None:
            n_dims = len(grid)
        if n_dims > 1 and options.DPDC:
            raise NotImplementedError()
        if n_threads is None:
            n_threads = numba.get_num_threads()

        self.__options = options
        self.__n_threads = 1 if n_dims == 1 else n_threads

        if self.__n_threads > 1:
            try:
                numba.parfors.parfor.ensure_parallel_support()
            except numba.core.errors.UnsupportedParforsError:
                print(
                    "Numba ensure_parallel_support() failed, forcing n_threads=1",
                    file=sys.stderr,
                )
                self.__n_threads = 1

        self.__n_dims = n_dims
        self.__call, self.traversals = make_step_impl(
            options,
            non_unit_g_factor,
            grid,
            self.n_threads,
            left_first=left_first,
            buffer_size=buffer_size,
        )

    @property
    def options(self) -> Options:
        """`Options` instance used"""
        return self.__options

    @property
    def n_threads(self) -> int:
        """actual n_threads used (may be different than passed to __init__ if n_dims==1
        or if on a platform where Numba does not support threading)"""
        return self.__n_threads

    @property
    def n_dims(self) -> int:
        """dimensionality (1, 2 or 3)"""
        return self.__n_dims

    def __call__(self, *, n_steps, mu_coeff, post_step, post_iter, fields):
        assert self.n_threads == 1 or numba.get_num_threads() == self.n_threads
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaExperimentalFeatureWarning)
            wall_time_per_timestep = self.__call(
                n_steps,
                mu_coeff,
                post_step,
                post_iter,
                *(
                    _Impl(field=v.impl[IMPL_META_AND_DATA], bc=v.impl[IMPL_BC])
                    for v in fields.values()
                ),
                self.traversals.data,
            )
        return wall_time_per_timestep


@lru_cache()
# pylint: disable=too-many-locals,too-many-statements,too-many-arguments
def make_step_impl(
    options, non_unit_g_factor, grid, n_threads, left_first, buffer_size
):
    """returns (and caches) an njit-ted stepping function and a traversals pair"""
    traversals = Traversals(
        grid=grid,
        halo=options.n_halo,
        jit_flags=options.jit_flags,
        n_threads=n_threads,
        left_first=left_first,
        buffer_size=buffer_size,
    )

    n_iters = options.n_iters
    non_zero_mu_coeff = options.non_zero_mu_coeff
    nonoscillatory = options.nonoscillatory

    upwind = make_upwind(options, non_unit_g_factor, traversals)
    flux_first_pass = make_flux_first_pass(options, traversals)
    flux_subsequent = make_flux_subsequent(options, traversals)
    antidiff = make_antidiff(non_unit_g_factor, options, traversals)
    antidiff_last_pass = make_antidiff(
        non_unit_g_factor, options, traversals, last_pass=True
    )
    laplacian = make_laplacian(non_unit_g_factor, options, traversals)
    nonoscillatory_psi_extrema = make_psi_extrema(options, traversals)
    nonoscillatory_beta = make_beta(non_unit_g_factor, options, traversals)
    nonoscillatory_correction = make_correction(options, traversals)
    axpy = make_axpy(options, traversals)

    @numba.njit(**options.jit_flags)
    # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,unnecessary-dunder-call
    def step(
        n_steps,
        mu_coeff,
        post_step,
        post_iter,
        advectee,
        advector,
        g_factor,
        vectmp_a,
        vectmp_b,
        vectmp_c,
        psi_extrema,
        beta,
        null_impl,
    ):
        time = clock()
        for step in range(n_steps):
            if non_zero_mu_coeff:
                advector_orig = advector
                advector = vectmp_c
            for iteration in range(n_iters):
                if iteration == 0:
                    if nonoscillatory:
                        nonoscillatory_psi_extrema(null_impl, psi_extrema, advectee)
                    if non_zero_mu_coeff:
                        laplacian(null_impl, advector, advectee)
                        axpy(
                            *advector.field,
                            mu_coeff,
                            *advector.field,
                            *advector_orig.field,
                        )
                    flux_first_pass(null_impl, vectmp_a, advector, advectee)
                    flux = vectmp_a
                else:
                    if iteration == 1:
                        advector_oscil = advector
                        advector_nonos = vectmp_a
                        flux = vectmp_b
                    elif iteration % 2 == 0:
                        advector_oscil = vectmp_a
                        advector_nonos = vectmp_b
                        flux = vectmp_a
                    else:
                        advector_oscil = vectmp_b
                        advector_nonos = vectmp_a
                        flux = vectmp_b
                    if iteration < n_iters - 1:
                        antidiff(
                            null_impl,
                            advector_nonos,
                            advectee,
                            advector_oscil,
                            g_factor,
                        )
                    else:
                        antidiff_last_pass(
                            null_impl,
                            advector_nonos,
                            advectee,
                            advector_oscil,
                            g_factor,
                        )
                    flux_subsequent(null_impl, flux, advectee, advector_nonos)
                    if nonoscillatory:
                        nonoscillatory_beta(
                            null_impl, beta, flux, advectee, psi_extrema, g_factor
                        )
                        # note: in libmpdata++, the oscillatory advector from prev iter is used
                        nonoscillatory_correction(null_impl, advector_nonos, beta)
                        flux_subsequent(null_impl, flux, advectee, advector_nonos)
                upwind(null_impl, advectee, flux, g_factor)
                post_iter.call(flux.field, g_factor.field, step, iteration)
            if non_zero_mu_coeff:
                advector = advector_orig
            post_step.call(advectee.field[ARG_DATA], step)
        return (clock() - time) / n_steps if n_steps > 0 else np.nan

    return step, traversals

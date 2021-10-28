""" MPDATA iteration logic """
from functools import lru_cache
import sys
import warnings
import numba
from numba.core.errors import NumbaExperimentalFeatureWarning
from .impl.formulae_upwind import make_upwind
from .impl.formulae_flux import make_flux_first_pass, make_flux_subsequent
from .impl.formulae_laplacian import make_laplacian
from .impl.formulae_antidiff import make_antidiff
from .impl.formulae_nonoscillatory import make_psi_extrema, make_beta, make_correction
from .impl.traversals import Traversals
from .impl.meta import META_HALO_VALID
from .impl.enumerations import INNER, MID3D, OUTER, ARG_DATA, IMPL_META_AND_DATA, IMPL_BC
from .options import Options
from .impl.clock import clock


class Stepper:
    """ MPDATA stepper specialised for given options, dimensionality and optionally grid
        (instances of Stepper can be shared among `Solver`s) """
    def __init__(self, *,
                 options: Options,
                 n_dims: (int, None) = None,
                 non_unit_g_factor: bool = False,
                 grid: (tuple, None) = None,
                 n_threads: (int, None) = None
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
                    file=sys.stderr)
                self.__n_threads = 1

        self.__n_dims = n_dims
        self.__call, self.traversals = make_step_impl(
            options, non_unit_g_factor, grid, self.n_threads
        )

    @property
    def options(self):
        return self.__options

    @property
    def n_threads(self):
        return self.__n_threads

    @property
    def n_dims(self):
        return self.__n_dims

    def __call__(self, n_steps, mu_coeff, post_step, post_iter, fields):
        assert self.n_threads == 1 or numba.get_num_threads() == self.n_threads
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
            wall_time_per_timestep = self.__call(
                n_steps, mu_coeff, post_step, post_iter,
                *(v.impl[i] for v in fields.values() for i in (IMPL_META_AND_DATA, IMPL_BC))
            )
        return wall_time_per_timestep


@lru_cache()
def make_step_impl(options, non_unit_g_factor, grid, n_threads):
    n_iters = options.n_iters
    n_dims = len(grid)
    halo = options.n_halo
    non_zero_mu_coeff = options.non_zero_mu_coeff
    nonoscillatory = options.nonoscillatory

    traversals = Traversals(grid, halo, options.jit_flags, n_threads=n_threads)

    upwind = make_upwind(options, non_unit_g_factor, traversals)
    flux_first_pass = make_flux_first_pass(options, traversals)
    flux_subsequent = make_flux_subsequent(options, traversals)
    antidiff = make_antidiff(non_unit_g_factor, options, traversals)
    antidiff_last_pass = make_antidiff(non_unit_g_factor, options, traversals, last_pass=True)
    laplacian = make_laplacian(non_unit_g_factor, options, traversals)
    nonoscillatory_psi_extrema = make_psi_extrema(options, traversals)
    nonoscillatory_beta = make_beta(non_unit_g_factor, options, traversals)
    nonoscillatory_correction = make_correction(options, traversals)

    @numba.njit(**options.jit_flags)
    # pylint: disable=too-many-arguments
    def axpy(out_meta, out_outer, out_mid3d, out_inner, a_coeffs,
             _, x_outer, x_mid3d, x_inner,
             __, y_outer, y_mid3d, y_inner):
        if n_dims > 1:
            out_outer[:] = a_coeffs[OUTER] * x_outer[:] + y_outer[:]
            if n_dims > 2:
                out_mid3d[:] = a_coeffs[MID3D] * x_mid3d[:] + y_mid3d[:]
        out_inner[:] = a_coeffs[INNER] * x_inner[:] + y_inner[:]
        out_meta[META_HALO_VALID] = False

    @numba.njit(**options.jit_flags)
    # pylint: disable=too-many-arguments
    def step(n_steps, mu_coeff, post_step, post_iter,
             advectee, advectee_bc,
             advector, advector_bc,
             g_factor, g_factor_bc,
             vectmp_a, _,
             vectmp_b, __,
             vectmp_c, ___,
             psi_extrema, psi_extrema_bc,
             beta, beta_bc
             ):
        vec_bc = advector_bc

        time = clock()
        for step in range(n_steps):
            if non_zero_mu_coeff:
                advector_orig = advector
                advector = vectmp_c
            for iteration in range(n_iters):
                if iteration == 0:
                    if nonoscillatory:
                        nonoscillatory_psi_extrema(psi_extrema, advectee, advectee_bc)
                    if non_zero_mu_coeff:
                        laplacian(advector, advectee, advectee_bc)
                        axpy(*advector, mu_coeff, *advector, *advector_orig)
                    flux_first_pass(vectmp_a, advector, advectee, advectee_bc, vec_bc)
                    flux = vectmp_a
                else:
                    if iteration == 1:
                        advector_oscilatory = advector
                        advector_nonoscilatory = vectmp_a
                        flux = vectmp_b
                    elif iteration % 2 == 0:
                        advector_oscilatory = vectmp_a
                        advector_nonoscilatory = vectmp_b
                        flux = vectmp_a
                    else:
                        advector_oscilatory = vectmp_b
                        advector_nonoscilatory = vectmp_a
                        flux = vectmp_b
                    if iteration < n_iters - 1:
                        antidiff(advector_nonoscilatory,
                                 advectee, advectee_bc,
                                 advector_oscilatory, vec_bc,
                                 g_factor, g_factor_bc)
                    else:
                        antidiff_last_pass(advector_nonoscilatory,
                                           advectee, advectee_bc,
                                           advector_oscilatory, vec_bc,
                                           g_factor, g_factor_bc)
                    flux_subsequent(flux, advectee, advectee_bc, advector_nonoscilatory, vec_bc)
                    if nonoscillatory:
                        nonoscillatory_beta(beta,
                                            flux, vec_bc,
                                            advectee, advectee_bc,
                                            psi_extrema, psi_extrema_bc,
                                            g_factor, g_factor_bc)
                        # note: in libmpdata++, the oscillatory advector from prev iter is used
                        nonoscillatory_correction(advector_nonoscilatory, vec_bc, beta, beta_bc)
                        flux_subsequent(flux, advectee, advectee_bc, advector_nonoscilatory, vec_bc)
                upwind(advectee, flux, vec_bc, g_factor, g_factor_bc)
                post_iter.__call__(flux, g_factor, step, iteration)
            if non_zero_mu_coeff:
                advector = advector_orig
            post_step.__call__(advectee[ARG_DATA], step)
        return (clock() - time) / n_steps
    return step, traversals

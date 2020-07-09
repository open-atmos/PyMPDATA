"""
Created at 20.03.2020
"""

import numba
from MPyDATA.formulae.upwind import make_upwind
from MPyDATA.formulae.flux import make_flux_first_pass, make_flux_subsequent
from MPyDATA.formulae.laplacian import make_laplacian
from MPyDATA.formulae.antidiff import make_antidiff
from MPyDATA.formulae.flux_corrected_transport import make_psi_extremum, make_beta, make_correction
from MPyDATA.arakawa_c.traversals import Traversals
from .arakawa_c.meta import meta_halo_valid
from MPyDATA.options import Options
from functools import lru_cache
from numba.core.errors import NumbaExperimentalFeatureWarning
import warnings
from .clock import clock


warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)


# import os
# os.environ["NUMBA_DISABLE_JIT"] = "1"


class Stepper:
    def __init__(self, *,
                 options: Options,
                 n_dims: (int, None) = None,
                 non_unit_g_factor: bool = False,
                 grid: (tuple, None) = None,
                 n_threads: (int, None) = None
                 ):
        self.options = options

        if n_dims is not None and grid is not None:
            raise ValueError()
        if n_dims is None and grid is None:
            raise ValueError()
        if grid is None:
            grid = tuple([-1] * n_dims)

        if n_threads is None:
            n_threads = numba.get_num_threads()
        self.n_threads = 1 if n_dims == 1 else n_threads
        if self.n_threads > 1 and numba.threading_layer() == 'workqueue':
            warnings.warn("Numba is using the ``workqueue'' threading layer, switch"
                          " to ``omp'' or ``tbb'' for higher parallel performance"
                          " (see https://numba.pydata.org/numba-doc/latest/user/threading-layer.html)")

        self.n_dims = n_dims
        self.__call = make_step_impl(options, non_unit_g_factor, grid, self.n_threads)

    def __call__(self, nt, mu_coeff,
                 advectee, advectee_bc,
                 advector, advector_bc,
                 g_factor, g_factor_bc,
                 vectmp_a, vectmp_a_bc,
                 vectmp_b, vectmp_b_bc,
                 vectmp_c, vectmp_c_bc,
                 psi_min, psi_min_bc,
                 psi_max, psi_max_bc,
                 beta_up, beta_up_bc,
                 beta_down, beta_down_bc):
        assert self.n_threads == 1 or numba.get_num_threads() == self.n_threads
        return self.__call(nt, mu_coeff,
                           advectee, advectee_bc,
                           advector, advector_bc,
                           g_factor, g_factor_bc,
                           vectmp_a, vectmp_a_bc,
                           vectmp_b, vectmp_b_bc,
                           vectmp_c, vectmp_c_bc,
                           psi_min, psi_min_bc,
                           psi_max, psi_max_bc,
                           beta_up, beta_up_bc,
                           beta_down, beta_down_bc)


@lru_cache()
def make_step_impl(options, non_unit_g_factor, grid, n_threads):
    n_iters = options.n_iters
    n_dims = len(grid)
    halo = options.n_halo
    non_zero_mu_coeff = options.non_zero_mu_coeff
    flux_corrected_transport = options.flux_corrected_transport

    traversals = Traversals(grid, halo, options.jit_flags, n_threads=n_threads)

    upwind = make_upwind(options, non_unit_g_factor, traversals)
    flux_first_pass = make_flux_first_pass(options, traversals)
    flux_subsequent = make_flux_subsequent(options, traversals)
    antidiff = make_antidiff(non_unit_g_factor, options, traversals)
    laplacian = make_laplacian(non_unit_g_factor, options, traversals)
    fct_psi_min = make_psi_extremum(min, options, traversals)
    fct_psi_max = make_psi_extremum(max, options, traversals)
    fct_beta_down = make_beta(min, non_unit_g_factor, options, traversals)
    fct_beta_up = make_beta(max, non_unit_g_factor, options, traversals)
    fct_correction = make_correction(options, traversals)

    @numba.njit(**options.jit_flags)
    def axpy(out_meta, out0, out1, a, x_meta, x0, x1, y_meta, y0, y1):
        out0[:] = a * x0[:] + y0[:]
        if n_dims > 1:
            out1[:] = a * x1[:] + y1[:]
        out_meta[meta_halo_valid] = False

    @numba.njit(**options.jit_flags)
    def step(nt, mu_coeff,
             advectee, advectee_bc,
             advector, advector_bc,
             g_factor, g_factor_bc,
             vectmp_a, vectmp_a_bc,
             vectmp_b, vectmp_b_bc,
             vectmp_c, vectmp_c_bc,
             psi_min, psi_min_bc,
             psi_max, psi_max_bc,
             beta_up, beta_up_bc,
             beta_down, beta_down_bc
             ):
        vec_bc = advector_bc
        null_vecfield = advector
        null_vecfield_bc = vec_bc

        time = clock()
        for _ in range(nt):
            if non_zero_mu_coeff:
                advector_orig = advector
                advector = vectmp_c
            for it in range(n_iters):
                if it == 0:
                    if flux_corrected_transport:
                        fct_psi_min(psi_min, advectee, advectee_bc, null_vecfield, null_vecfield_bc)
                        fct_psi_max(psi_max, advectee, advectee_bc, null_vecfield, null_vecfield_bc)
                    if non_zero_mu_coeff:
                        laplacian(advector, advectee, advectee_bc, null_vecfield_bc)
                        axpy(*advector, mu_coeff, *advector, *advector_orig)
                    flux_first_pass(vectmp_a, advector, advectee, advectee_bc, vec_bc)
                    flux = vectmp_a
                else:
                    if it == 1:
                        advector_oscilatory = advector
                        advector_nonoscilatory = vectmp_a
                        flux = vectmp_b
                    elif it % 2 == 0:
                        advector_oscilatory = vectmp_a
                        advector_nonoscilatory = vectmp_b
                        flux = vectmp_a
                    else:
                        advector_oscilatory = vectmp_b
                        advector_nonoscilatory = vectmp_a
                        flux = vectmp_b
                    antidiff(advector_nonoscilatory, advectee, advectee_bc, advector_oscilatory, vec_bc, g_factor, g_factor_bc)
                    flux_subsequent(flux, advectee, advectee_bc, advector_nonoscilatory, vec_bc)
                    if flux_corrected_transport:
                        fct_beta_down(beta_down, flux, vec_bc, advectee, advectee_bc, psi_min, psi_min_bc, g_factor, g_factor_bc)
                        fct_beta_up(beta_up, flux, vec_bc, advectee, advectee_bc, psi_max, psi_max_bc, g_factor, g_factor_bc)
                        fct_correction(advector_nonoscilatory, vec_bc, beta_down, beta_down_bc, beta_up, beta_up_bc)
                        flux_subsequent(flux, advectee, advectee_bc, advector_nonoscilatory, vec_bc)
                upwind(advectee, flux, vec_bc, g_factor, g_factor_bc)
            if non_zero_mu_coeff:
                advector = advector_orig
        return (clock() - time) / nt
    return step

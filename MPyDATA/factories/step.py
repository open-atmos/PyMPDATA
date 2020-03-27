"""
Created at 20.03.2020
"""

import numba
from MPyDATA.jit_flags import jit_flags
from MPyDATA.formulae.upwind import make_upwind
from MPyDATA.formulae.flux import make_flux_first_pass, make_flux_subsequent
from MPyDATA.formulae.laplacian import make_laplacian
from MPyDATA.formulae.antidiff import make_antidiff
from MPyDATA.formulae.flux_corrected_transport import make_psi_extremum, make_beta, make_correction
from MPyDATA.arakawa_c.traversals import Traversals


def make_step(*,
              options,
              grid,
              halo,
              non_unit_g_factor=False
              ):

    n_dims = len(grid)
    n_iters = options.n_iters
    mu_coeff = options.mu_coeff
    flux_corrected_transport = options.flux_corrected_transport

    traversals = Traversals(grid, halo)

    upwind = make_upwind(non_unit_g_factor, traversals)

    flux_first_pass = make_flux_first_pass(traversals)
    flux_subsequent = make_flux_subsequent(options, traversals)
    antidiff = make_antidiff(non_unit_g_factor, options, traversals)
    laplacian = make_laplacian(non_unit_g_factor, options, traversals)
    fct_psi_min = make_psi_extremum(min, options, traversals)
    fct_psi_max = make_psi_extremum(max, options, traversals)
    fct_beta_down = make_beta(min, non_unit_g_factor, options, traversals)
    fct_beta_up = make_beta(max, non_unit_g_factor, options, traversals)
    fct_correction = make_correction(options, traversals)

    @numba.njit(**jit_flags)
    def add(af, a0, a1, bf, b0, b1):
        b0[:] += a0[:]
        if n_dims > 1:
            b1[:] += a1[:]
        bf[0] = False

    @numba.njit(**jit_flags)
    def step(nt,
             psi, psi_bc,
             GC_phys, GC_phys_bc,
             g_factor, g_factor_bc,
             vectmp_a, vectmp_a_bc,
             vectmp_b, vectmp_b_bc,
             vectmp_c, vectmp_c_bc,
             psi_min, psi_min_bc,
             psi_max, psi_max_bc,
             beta_up, beta_up_bc,
             beta_down, beta_down_bc
             ):
        vec_bc = GC_phys_bc
        null_vecfield = GC_phys
        null_vecfield_bc = vec_bc

        for _ in range(nt):
            if mu_coeff != 0:
                GC_orig = GC_phys
                GC_phys = vectmp_c
            for it in range(n_iters):
                if it == 0:
                    if flux_corrected_transport:
                        fct_psi_min(psi_min, psi, psi_bc, null_vecfield, null_vecfield_bc)
                        fct_psi_max(psi_max, psi, psi_bc, null_vecfield, null_vecfield_bc)
                    if mu_coeff != 0:
                        laplacian(GC_phys, psi, psi_bc, null_vecfield_bc)
                        add(*GC_orig, *GC_phys)
                    flux_first_pass(vectmp_a, GC_phys, psi, psi_bc, vec_bc)
                    flux = vectmp_a
                else:
                    if it == 1:
                        GC_unco = GC_phys
                        GC_corr = vectmp_a
                        flux = vectmp_b
                    elif it % 2 == 0:
                        GC_unco = vectmp_a
                        GC_corr = vectmp_b
                        flux = vectmp_a
                    else:
                        GC_unco = vectmp_b
                        GC_corr = vectmp_a
                        flux = vectmp_b
                    antidiff(GC_corr, psi, psi_bc, GC_unco, vec_bc, g_factor, g_factor_bc)
                    flux_subsequent(flux, psi, psi_bc, GC_corr, vec_bc)
                    if flux_corrected_transport:
                        fct_beta_down(beta_down, flux, vec_bc, psi, psi_bc, psi_min, psi_min_bc, g_factor, g_factor_bc)
                        fct_beta_up(beta_up, flux, vec_bc, psi, psi_bc, psi_max, psi_max_bc, g_factor, g_factor_bc)
                        fct_correction(GC_corr, vec_bc, beta_down, beta_down_bc, beta_up, beta_up_bc)
                        flux_subsequent(flux, psi, psi_bc, GC_corr, vec_bc)
                upwind(psi, flux, vec_bc, g_factor, g_factor_bc)
    return step

"""
Created at 20.03.2020
"""

import numba
from MPyDATA.jit_flags import jit_flags
from MPyDATA.formulae.upwind import make_upwind
from MPyDATA.formulae.flux import make_flux_first_pass, make_flux_subsequent
from MPyDATA.formulae.laplacian import make_laplacian
from MPyDATA.formulae.antidiff import make_antidiff
from MPyDATA.arakawa_c.traversals import make_traversals


def make_step(*,
              options,
              grid,
              halo,
              non_unit_g_factor=False,
              mu_coeff=0,
              ):

    n_dims = len(grid)
    n_iters = options.n_iters

    apply_scalar, apply_vector = make_traversals(grid, n_dims, halo)

    upwind = make_upwind(n_dims, non_unit_g_factor, apply_scalar)

    flux_first_pass = make_flux_first_pass(n_dims, apply_vector)
    flux_subsequent = make_flux_subsequent(n_dims, options, apply_vector)
    antidiff = make_antidiff(n_dims, non_unit_g_factor, options, apply_vector)
    laplacian = make_laplacian(n_dims, non_unit_g_factor, options, apply_vector)

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
             vectmp_c, vectmp_c_bc
             ):
        vec_bc = GC_phys_bc

        for _ in range(nt):
            if mu_coeff != 0:
                GC_orig = GC_phys
                GC_phys = vectmp_c
            for it in range(n_iters):
                if it == 0:
                    if mu_coeff != 0:
                        laplacian(GC_phys, psi, psi_bc, vec_bc)
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
                upwind(psi, flux, vec_bc, g_factor, g_factor_bc)
    return step

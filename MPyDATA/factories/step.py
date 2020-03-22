"""
Created at 20.03.2020

@author: Piotr Bartman
"""

import numba
from MPyDATA.formulae.jit_flags import jit_flags
from MPyDATA.formulae.upwind import make_upwind
from MPyDATA.formulae.flux import make_flux_first_pass, make_flux_subsequent
from MPyDATA.formulae.laplacian import make_laplacian
from MPyDATA.formulae.antidiff import make_antidiff
from MPyDATA.arakawa_c.utils import indexers
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
    idx = indexers[n_dims]

    apply_scalar, apply_vector = make_traversals(grid, n_dims, halo)

    @numba.njit(**jit_flags)
    def null_formula(_, __, ___):
        return 44.

    formulae_flux_first_pass = (
        make_flux_first_pass(idx.atv0, idx.at0),
        make_flux_first_pass(idx.atv1, idx.at1),
        null_formula,
        null_formula
    )

    formulae_upwind = (
        make_upwind(idx.atv0, idx.at0, non_unit_g_factor),
        make_upwind(idx.atv1, idx.at1, non_unit_g_factor)
    )

    if n_iters > 1:
        formulae_flux_subsequent = (
            make_flux_subsequent(idx.atv0, idx.at0, infinite_gauge=options.infinite_gauge),
            make_flux_subsequent(idx.atv1, idx.at1, infinite_gauge=options.infinite_gauge),
            null_formula,
            null_formula
        )

        formulae_antidiff = (
            make_antidiff(idx.atv0, idx.at0, non_unit_g_factor=non_unit_g_factor, options=options, n_dims=n_dims, axis=0),
            make_antidiff(idx.atv1, idx.at1, non_unit_g_factor=non_unit_g_factor, options=options, n_dims=n_dims, axis=0),
            make_antidiff(idx.atv0, idx.at0, non_unit_g_factor=non_unit_g_factor, options=options, n_dims=n_dims, axis=1),
            make_antidiff(idx.atv1, idx.at1, non_unit_g_factor=non_unit_g_factor, options=options, n_dims=n_dims, axis=1)
        )
    else:
        formulae_flux_subsequent = (null_formula, null_formula, null_formula, null_formula)
        formulae_antidiff = (null_formula, null_formula, null_formula, null_formula)

    if mu_coeff == 0:
        formulae_laplacian = (null_formula, null_formula, null_formula, null_formula)
    else:
        formulae_laplacian = (
            make_laplacian(idx.at0, mu_coeff, options.epsilon, non_unit_g_factor, n_dims),
            make_laplacian(idx.at1, mu_coeff, options.epsilon, non_unit_g_factor, n_dims),
            null_formula,
            null_formula
        )

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
        # TODO
        null_vecfield = GC_phys
        null_scalarfield = psi
        null_bc = GC_phys_bc

        vec_bc = GC_phys_bc

        for _ in range(nt):
            if mu_coeff != 0:
                GC_orig = GC_phys
                GC_phys = vectmp_c
            for it in range(n_iters):
                if it == 0:
                    if mu_coeff != 0:
                        apply_vector(False, *formulae_laplacian, *GC_phys, *psi, *psi_bc, *null_vecfield, *null_bc, *null_scalarfield, *null_bc)
                        add(*GC_orig, *GC_phys)
                    apply_vector(False, *formulae_flux_first_pass, *vectmp_a, *psi, *psi_bc, *GC_phys, *vec_bc, *null_scalarfield, *null_bc)
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
                    apply_vector(True, *formulae_antidiff, *GC_corr, *psi, *psi_bc, *GC_unco, *vec_bc, *g_factor, *g_factor_bc)
                    apply_vector(False, *formulae_flux_subsequent, *flux, *psi, *psi_bc, *GC_corr, *vec_bc, *null_scalarfield, *null_bc)
                apply_scalar(*formulae_upwind, *psi, *flux, *vec_bc, *g_factor, *g_factor_bc)
    return step

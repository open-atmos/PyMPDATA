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
from MPyDATA.arakawa_c.utils import at_1d, at_2d_axis0, at_2d_axis1, atv_1d, atv_2d_axis0, atv_2d_axis1
from MPyDATA.arakawa_c.boundary_condition.cyclic import Cyclic
from MPyDATA.arakawa_c.traversals import make_traversals


def make_step(*,
              options,
              grid,
              halo,
              non_unit_g_factor=False,
              mu_coeff=0,
              boundary_conditions=(Cyclic, Cyclic)
              ):

    n_dims = len(grid)
    n_iters = options.n_iters

    if n_dims == 1:
        at0 = at_1d
        at1 = at_1d  # TODO: redundant
        atv0 = atv_1d
        atv1 = atv_1d  # TODO: redundant
    elif n_dims == 2:
        at0 = at_2d_axis0
        at1 = at_2d_axis1
        atv0 = atv_2d_axis0
        atv1 = atv_2d_axis1
    else:
        raise NotImplementedError()

    fill_halos = (
        boundary_conditions[0].make(at0, halo),
        boundary_conditions[1].make(at1, halo) if n_dims > 1 else None
    )
    apply_scalar, apply_vector = make_traversals(grid, n_dims, halo, fill_halos)

    @numba.njit(**jit_flags)
    def null_formula(_, __):
        return 44.

    formulae_flux_first_pass = (
        make_flux_first_pass(atv0, at0),
        make_flux_first_pass(atv1, at1),
        null_formula,
        null_formula
    )

    formulae_upwind = (
        make_upwind(atv0, at0, non_unit_g_factor),
        make_upwind(atv1, at1, non_unit_g_factor)
    )

    formulae_flux_subsequent = (
        make_flux_subsequent(atv0, at0, infinite_gauge=options.infinite_gauge),
        make_flux_subsequent(atv1, at1, infinite_gauge=options.infinite_gauge),
        null_formula,
        null_formula
    )

    formulae_antidiff = (
        make_antidiff(atv0, at0, infinite_gauge=options.infinite_gauge, epsilon=options.epsilon, n_dims=n_dims, axis=0),
        make_antidiff(atv1, at1, infinite_gauge=options.infinite_gauge, epsilon=options.epsilon, n_dims=n_dims, axis=0),
        make_antidiff(atv0, at0, infinite_gauge=options.infinite_gauge, epsilon=options.epsilon, n_dims=n_dims, axis=1),
        make_antidiff(atv1, at1, infinite_gauge=options.infinite_gauge, epsilon=options.epsilon, n_dims=n_dims, axis=1)
    )

    if mu_coeff == 0:
        formulae_laplacian = (null_formula, null_formula, null_formula, null_formula)
    else:
        formulae_laplacian = (
            make_laplacian(at0, mu_coeff, options.epsilon, non_unit_g_factor, n_dims),
            make_laplacian(at1, mu_coeff, options.epsilon, non_unit_g_factor, n_dims),
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
    def copy(af, a0, a1, bf, b0, b1):
        b0[:] = a0[:]
        if n_dims > 1:
            b1[:] = a1[:]
        bf[0] = False

    @numba.njit(**jit_flags)
    def step(nt, psi, GC_phys, g_factor, vectmp_a, vectmp_b, vectmp_c):
        # TODO
        null_vecfield = GC_phys

        for _ in range(nt):
            if mu_coeff != 0:
                GC_orig = GC_phys
                GC_phys = vectmp_c
            for it in range(n_iters):
                if it == 0:
                    if mu_coeff != 0:
                        apply_vector(False, *formulae_laplacian, *GC_phys, *psi, *null_vecfield)
                        add(*GC_orig, *GC_phys)
                    apply_vector(False, *formulae_flux_first_pass, *vectmp_a, *psi, *GC_phys)
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
                    apply_vector(True, *formulae_antidiff, *GC_corr, *psi, *GC_unco)
                    apply_vector(False, *formulae_flux_subsequent, *flux, *psi, *GC_corr)
                apply_scalar(*formulae_upwind, *psi, *flux, g_factor)
    return step

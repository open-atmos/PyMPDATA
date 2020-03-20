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


def make_step(options, grid, halo, non_unit_g_factor=False, mu_coeff=0, boundary_condition=Cyclic):

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

    fill_halos = (boundary_condition.make(at0), boundary_condition.make(at1))
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
        make_antidiff(atv0, at0, infinite_gauge=options.infinite_gauge, epsilon=options.epsilon, n_dims=n_dims, axis=1),
        make_antidiff(atv1, at1, infinite_gauge=options.infinite_gauge, epsilon=options.epsilon, n_dims=n_dims, axis=0),
        make_antidiff(atv1, at1, infinite_gauge=options.infinite_gauge, epsilon=options.epsilon, n_dims=n_dims, axis=1)
    )

    if mu_coeff == 0:
        laplacian = (null_formula, null_formula)
    else:
        formulae_laplacian = (
            make_laplacian(at0, mu_coeff, options.epsilon, non_unit_g_factor, n_dims),
            make_laplacian(at1, mu_coeff, options.epsilon, non_unit_g_factor, n_dims)
        )

    formulae_laplacian = (laplacian, laplacian)

    @numba.njit(**jit_flags)
    def step(nt, psi, GC_phys, g_factor, vectmp_a, vectmp_b, vectmp_c):
        flux = vectmp_a
        GC_orig = vectmp_c  # only for mu_coeff != 0

        # TODO
        null_vecfield = GC_phys

        for _ in range(nt):
            if mu_coeff != 0:
                for d in range(n_dims):
                    GC_orig[d][:] = GC_phys[d][:]
            for it in range(n_iters):
                if it == 0:
                    if mu_coeff != 0:
                        apply_vector(False, *formulae_laplacian, *GC_phys, *psi, *null_vecfield)
                        for d in range(n_dims):
                            GC_phys[d][:] += GC_orig[d][:]
                    apply_vector(False, *formulae_flux_first_pass, *flux, *psi, *GC_phys)
                else:
                    if it == 1:
                        apply_vector(True, *formulae_antidiff, *vectmp_b, *psi, *GC_phys)
                        apply_vector(False, *formulae_flux_subsequent, *flux, *psi, *vectmp_b)
                    else:
                        apply_vector(True, *formulae_antidiff, *vectmp_a, *psi, *vectmp_b)
                        apply_vector(False, *formulae_flux_subsequent, *flux, *psi, *vectmp_a)
                apply_scalar(*formulae_upwind, *psi, *flux, g_factor)
    return step

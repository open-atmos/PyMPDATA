"""
Created at 20.03.2020
"""

import numba
from MPyDATA.jit_flags import jit_flags
from MPyDATA.arakawa_c.utils import indexers, null_formula


def make_laplacian(non_unit_g_factor, options, traversals):
    if options.mu_coeff == 0:
        @numba.njit(**jit_flags)
        def apply(_flux, _psi, _psi_bc, _GC_corr, _vec_bc):
            return
    else:
        idx = indexers[traversals.n_dims]
        apply_vector = traversals.apply_vector()

        formulae_laplacian = (*tuple([
            __make_laplacian(idx.at[i], options.mu_coeff, options.epsilon, non_unit_g_factor, traversals.n_dims)
            for i in range(2)]),
        )

        @numba.njit(**jit_flags)
        def apply(GC_phys, psi, psi_bc, null_vecfield_bc):
            null_vecfield = GC_phys
            null_scalarfield = psi
            return apply_vector(*formulae_laplacian, *GC_phys, *psi, *psi_bc, *null_vecfield, *null_vecfield_bc,
                                *null_scalarfield, *null_vecfield_bc)

    return apply


def __make_laplacian(at, mu, epsilon, non_unit_g_factor, n_dims):
    if non_unit_g_factor or n_dims > 1:
        raise NotImplementedError()

    @numba.njit(**jit_flags)
    def A(psi, _, __):
        return -2 * mu * (
                at(*psi, 1, 0) - at(*psi, 0, 0)
        ) / (
                at(*psi, 1, 0) + at(*psi, 0, 0) + epsilon
        )
    return A

import numba
from MPyDATA.jit_flags import jit_flags
from MPyDATA.arakawa_c.utils import indexers, null_formula

def make_laplacian(n_dims, non_unit_g_factor, options, apply_vector):
    if options.mu_coeff == 0:
        @numba.njit(**jit_flags)
        def apply(_flux, _psi, _psi_bc, _GC_corr, _vec_bc):
            return
    else:
        idx = indexers[n_dims]

        formulae_laplacian = (*tuple([
            __make_laplacian(idx.at[i], options.mu_coeff, options.epsilon, non_unit_g_factor, n_dims)
            for i in range(2)]),
            null_formula,
            null_formula
        )

        @numba.njit(**jit_flags)
        def apply(GC_phys, psi, psi_bc, vec_bc):
            null_vecfield = GC_phys
            null_scalarfield = psi
            null_bc = vec_bc  # faster
            return apply_vector(False, *formulae_laplacian, *GC_phys, *psi, *psi_bc, *null_vecfield, *null_bc,
                                *null_scalarfield, *null_bc)

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
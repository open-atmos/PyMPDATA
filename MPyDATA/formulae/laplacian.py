import numba
from MPyDATA.jit_flags import jit_flags


def make_laplacian(at, mu, epsilon, non_unit_g_factor, n_dims):
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
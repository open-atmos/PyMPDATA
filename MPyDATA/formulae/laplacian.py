from ..utils import debug_flag

if debug_flag.VALUE:
    import MPyDATA.utils.fake_numba as numba
else:
    import numba


def make_laplacian(opts):
    if not opts.nzm:
        return

    eps = opts.eps

    @numba.njit
    def A(psi, mu):
        result = -2 * mu.value * (
                psi.at(1, 0) - psi.at(0, 0)
        ) / (
                psi.at(1, 0) + psi.at(0, 0) + eps
        )
        return result
    return A

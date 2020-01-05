from ..options import Options
from ..utils import debug_flag

if debug_flag.VALUE:
    import MPyDATA.utils.fake_numba as numba
else:
    import numba


def make_laplacian(opts: Options):
    if opts.mu == 0:
        return

    eps = opts.eps
    mu = opts.mu

    @numba.njit
    def A(psi):
        result = -2 * mu * (
                psi.at(1, 0) - psi.at(0, 0)
        ) / (
                psi.at(1, 0) + psi.at(0, 0) + eps
        )
        return result
    return A

from ..arakawa_c.traversal import Traversal
from ..utils import debug_flag
from .jit_flags import jit_flags

if debug_flag.VALUE:
    import MPyDATA.utils.fake_numba as numba
else:
    import numba


def make_laplacian(opts):
    if not opts.nzm:
        return

    eps = opts.eps

    @numba.njit(**jit_flags)
    def A(init: float, psi, mu):
        result = -2 * mu.value * (
                psi.at(1, 0) - psi.at(0, 0)
        ) / (
                psi.at(1, 0) + psi.at(0, 0) + eps
        )
        return init + result
    return Traversal(body=A, init=0, loop=True)

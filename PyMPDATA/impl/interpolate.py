"""Interpolation formulae sketch"""

import numba
import numpy as np


def make_interpolate(options):
    """Function that returns JIT-compilable interpolate function"""

    @numba.njit(**options.jit_flags)
    def interpolate(psi, axis):
        idx = (
            (slice(None, -1), slice(None, None)),
            (slice(None, None), slice(None, -1)),
        )
        s1 = 2 * [slice(None)]
        s2 = 2 * [slice(None)]
        s1[axis] = slice(1, None)
        s2[axis] = slice(None, -1)
        s1 = tuple(s1)
        s2 = tuple(s2)
        out = psi[s1] - psi[s2]
        return out / 2 * psi[idx[axis]]

    return interpolate

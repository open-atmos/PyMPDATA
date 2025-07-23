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
        return np.diff(psi, axis=axis) / 2 * psi[idx[axis]]

    return interpolate

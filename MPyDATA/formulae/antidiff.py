import numba
import numpy as np
from .jit_flags import jit_flags


def make_antidiff(atv, at, infinite_gauge, epsilon, n_dims, axis):
    # eq. 13 in Smolarkiewicz 1984; eq. 17a in Smolarkiewicz & Margolin 1998
    @numba.njit(**jit_flags)
    def A(psi):
        result = at(*psi, 1, 0) - at(*psi, 0, 0)
        if infinite_gauge:
            result /= 2
        else:
            result /= (at(*psi, 1, 0) + at(*psi, 0, 0) + epsilon)
        return result

    # eq. 13 in Smolarkiewicz 1984; eq. 17b in Smolarkiewicz & Margolin 1998
    @numba.njit(**jit_flags)
    def B(psi):
        result = (
                at(*psi, 1, 1) + at(*psi, 0, 1) -
                at(*psi, 1, -1) - at(*psi, 0, -1)
        )
        if infinite_gauge:
            result /= 4
        else:
            result /= (
                    at(*psi, 1, 1) + at(*psi, 0, 1) +
                    at(*psi, 1, -1) + at(*psi, 0, -1) +
                    epsilon
            )
        return result

    @numba.njit(**jit_flags)
    def antidiff(psi, GC):
        # eq. 13 in Smolarkiewicz 1984
        result = (np.abs(atv(*GC, .5, 0)) - atv(*GC, +.5, 0) ** 2) * A(psi)
        if axis == 0 and n_dims == 1:
            return result
        else:
            result -= (
                0.5 * atv(*GC, .5, 0) *
                0.25 * (atv(*GC, 1, +.5) + atv(*GC, 0, +.5) + atv(*GC, 1, -.5) + atv(*GC, 0, -.5)) *
                B(psi)
            )

        # TODO: tot
        # TODO: dfl

        return result
    return antidiff

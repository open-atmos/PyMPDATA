import numba
import numpy as np
from .jit_flags import jit_flags


def make_antidiff(atv, at, infinite_gauge, epsilon, n_dims):
    @numba.njit(**jit_flags)
    def antidiff(axis, psi, GC):
        # eq. 13 in Smolarkiewicz 1984; eq. 17a in Smolarkiewicz & Margolin 1998
        def A(psi):
            result = at(*psi, 1, 0) - at(*psi, 0, 0)
            if infinite_gauge:
                result /= 2
            else:
                result /= (at(*psi, 1, 0) + at(*psi, 0, 0) + epsilon)
            return result

        # eq. 13 in Smolarkiewicz 1984; eq. 17b in Smolarkiewicz & Margolin 1998
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

        # eq. 13 in Smolarkiewicz 1984
        result = (np.abs(atv(*GC, .5, 0)) - atv(*GC, +.5, 0) ** 2) * A(psi)
        for i in range(n_dims):
            if i == axis:
                continue
            result -= (
                    0.5 * atv(*GC, +.5, 0) *
                    0.25 * (atv(*GC, 1, +.5) + atv(*GC, 0, +.5) + atv(*GC, 1, -.5) + atv(*GC, 0, -.5)) *
                    B(psi)
            )
        return result
    return antidiff


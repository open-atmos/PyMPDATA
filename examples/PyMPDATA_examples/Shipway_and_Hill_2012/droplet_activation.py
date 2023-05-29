from functools import lru_cache
import numpy as np
import numba
from PyMPDATA.impl.enumerations import (SIGN_RIGHT,
                                        ARG_DATA, OUTER, ARG_FOCUS,
                                        META_AND_DATA_DATA, META_AND_DATA_META)
from PyMPDATA.impl.traversals_common import make_fill_halos_loop


@lru_cache()
def _make_scalar(value, set_value, halo, dtype, jit_flags, _):
    @numba.njit(**jit_flags)
    def impl(psi, __, sign):
        if sign == SIGN_RIGHT:
            return 0
        z = psi[ARG_FOCUS][OUTER]
        activated = np.sum(psi[ARG_DATA][z:z+1, halo:-halo])
        # assert activated < value
        result = max(0, value - activated)
        return result

    if dtype == complex:
        @numba.njit(**jit_flags)
        def fill_halos_scalar(psi, n, sign):
            return complex(
                impl((psi[META_AND_DATA_META], psi[META_AND_DATA_DATA].real), n, sign),
                impl((psi[META_AND_DATA_META], psi[META_AND_DATA_DATA].imag), n, sign)
            )
    else:
        @numba.njit(**jit_flags)
        def fill_halos_scalar(psi, n, sign):
            return impl(psi, n, sign)
    return make_fill_halos_loop(jit_flags, set_value, fill_halos_scalar)


# pylint: disable=too-few-public-methods
class DropletActivation:
    def __init__(self, value, dr, dz):
        self._value = value / dz / dr

    def make_scalar(self, indexers, halo, dtype, jit_flags, dimension_index):
        return _make_scalar(self._value, indexers.set, halo, dtype, jit_flags, dimension_index)

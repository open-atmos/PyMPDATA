""" boundary condition filling halos with a constant value """
from functools import lru_cache
import numba


class Constant:
    """ class which instances are to be passed in boundary_conditions tuple to the
        `ScalarField` and `VectorField` __init__ methods """
    def __init__(self, value):
        self.value = value

    def make_scalar(self, ats, halo, dtype, jit_flags):
        """ returns (lru-cached) Numba-compiled scalar halo-filling callable """
        return _make_scalar(self.value, ats, halo, dtype, jit_flags)

    def make_vector(self, ats, halo, dtype, jit_flags):
        """ returns (lru-cached) Numba-compiled vector halo-filling callable """
        return _make_scalar(self.value, ats, halo, dtype, jit_flags)


@lru_cache()
def _make_scalar(value, _, __, ___, jit_flags):
    @numba.njit(**jit_flags)
    def fill_halos(_1, _2, _3):
        return value

    return fill_halos

import numba
from functools import lru_cache


@lru_cache()
def _make_scalar(value, at, halo):

    @numba.njit()
    def fill_halos(_, __, ___):
        return value

    return fill_halos


class ConstantBoundaryCondition:
    def __init__(self, value):
        self._value = value

    def make_scalar(self, _at, _halo):
        return _make_scalar(self._value, _at, _halo)

    def make_vector(self, at):
        return ConstantBoundaryCondition(self._value).make_scalar(at, 0)

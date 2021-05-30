import numba
from ..enumerations import SIGN_RIGHT, SIGN_LEFT
from functools import lru_cache


@lru_cache()
def _make_scalar(at):
    @numba.njit()
    def fill_halos(psi, n, sign):
        return at(*psi, sign * n)
    return fill_halos


@lru_cache()
def _make_vector(at):
    @numba.njit()
    def fill_halos(psi, n, sign):
        return at(*psi, sign * n)
    return fill_halos


class PeriodicBoundaryCondition:
    def __init__(self):
        assert SIGN_RIGHT == -1
        assert SIGN_LEFT == +1

    def make_scalar(self, at, _):
        return _make_scalar(at)

    def make_vector(self, at):
        return _make_vector(at)

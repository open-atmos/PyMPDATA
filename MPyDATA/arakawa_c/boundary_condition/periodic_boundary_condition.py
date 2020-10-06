"""
Created at 20.03.2020

@author: Piotr Bartman
"""

import numba
from functools import lru_cache


@lru_cache()
def _make_scalar(at):
    @numba.njit()
    def fill_halos(psi, n, sign):
        return at(*psi, sign * n, 0)
    return fill_halos


@lru_cache()
def _make_vector(at):
    @numba.njit()
    def fill_halos(psi, n, sign):
        return at(*psi, sign * n, 0)
    return fill_halos


class PeriodicBoundaryCondition:
    def make_scalar(self, at, _):
        return _make_scalar(at)

    def make_vector(self, at):
        return _make_vector(at)

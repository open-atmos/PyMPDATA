"""
Created at 20.03.2020

@author: Piotr Bartman
"""

import numba
from ..enumerations import SIGN_RIGHT, SIGN_LEFT


class PeriodicBoundaryCondition:
    def __init__(self):
        assert SIGN_RIGHT == -1
        assert SIGN_LEFT == +1

    def make_scalar(self, at, _):
        @numba.njit()
        def fill_halos(psi, n, sign):
            return at(*psi, sign * n, 0, 0)
        return fill_halos

    def make_vector(self, at):
        @numba.njit()
        def fill_halos(psi, n, sign):
            return at(*psi, sign * n, 0, 0)
        return fill_halos

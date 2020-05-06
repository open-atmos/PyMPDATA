"""
Created at 20.03.2020

@author: Piotr Bartman
"""

import numba


class PeriodicBoundaryCondition:
    def make_scalar(self, at, _):
        @numba.njit()
        def fill_halos(psi, n, sign):
            return at(*psi, sign * n, 0)
        return fill_halos

    def make_vector(self, at):
        @numba.njit()
        def fill_halos(psi, n, sign):
            return at(*psi, sign * n, 0)
        return fill_halos

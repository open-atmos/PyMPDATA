"""
Created at 20.03.2020

@author: Piotr Bartman
"""

import numba
from MPyDATA.jit_flags import jit_flags


class Cyclic:
    @staticmethod
    def make_scalar(at, _):
        @numba.njit(**jit_flags)
        def fill_halos(psi, n, sign):
            return at(*psi, sign * n, 0)
        return fill_halos

    @staticmethod
    def make_vector(at):
        @numba.njit(**jit_flags)
        def fill_halos(psi, n, sign):
            return at(*psi, sign * n, 0)
        return fill_halos

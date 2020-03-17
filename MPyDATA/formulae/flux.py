"""
Created at 11.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np
from .jit_flags import jit_flags
import numba


@numba.njit(**jit_flags)
def minimum_0(c):
    return (c - np.abs(c)) / 2


@numba.njit(**jit_flags)
def maximum_0(c):
    return (c + np.abs(c)) / 2


def make_flux(atv, at):
    @numba.njit(**jit_flags)
    def flux(psi, GC):
        return \
            maximum_0(atv(*GC, +.5, 0)) * at(*psi, 0, 0) + \
            minimum_0(atv(*GC, +.5, 0)) * at(*psi, 1, 0)
    return flux

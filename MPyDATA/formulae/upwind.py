"""
Created at 11.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from .jit_flags import jit_flags
import numba


def make_upwind(atv, at, nug):
    @numba.njit(**jit_flags)
    def upwind(focus, init,
               flux, g_factor):
        result = \
               + atv(focus, flux, -.5, 0) \
               - atv(focus, flux, .5, 0)
        if nug:
            result /= at(focus, g_factor, 0, 0)
        return init + result
    return upwind

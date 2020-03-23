"""
Created at 11.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from MPyDATA.jit_flags import jit_flags
import numba


def make_upwind(atv, at, nug):
    @numba.njit(**jit_flags)
    def upwind(init, flux, g_factor):
        result = \
               + atv(*flux, -.5, 0) \
               - atv(*flux, .5, 0)
        if nug:
            result /= at(*g_factor, 0, 0)
        return init + result
    return upwind

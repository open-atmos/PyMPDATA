"""
Created at 11.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from .jit_flags import jit_flags
import numba


def make_upwind(atv):
    @numba.njit(**jit_flags)
    def upwind(focus, flux_tpl, init):
        return init \
               + atv(focus, flux_tpl, -.5, 0) \
               - atv(focus, flux_tpl, .5, 0)
    return upwind

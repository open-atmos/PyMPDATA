"""
Created at 11.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from ..arakawa_c.vector_field import VectorField
from ..arakawa_c.impl.vector_field_2d import at
from ..utils import debug_flag
from .jit_flags import jit_flags

if debug_flag.VALUE:
    import MPyDATA.utils.fake_numba as numba
else:
    import numba


def make_upwind():
    @numba.njit(**jit_flags)
    def upwind(init: float, _data_0, _data_1, arg1_i, arg1_j, dim):
        result = -1 * (
                at(_data_0, _data_1, arg1_i, arg1_j, dim, +.5, 0) -
                at(_data_0, _data_1, arg1_i, arg1_j, dim, -.5, 0)
        )
        return init + result
    return upwind, 0, True

#
# @numba.stencil()
# def upwind()

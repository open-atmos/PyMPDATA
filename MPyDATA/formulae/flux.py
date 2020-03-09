"""
Created at 11.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from ..arakawa_c.impl import vector_field_2d
from ..arakawa_c.impl import scalar_field_2d
import numpy as np
from ..utils import debug_flag
from .jit_flags import jit_flags

if debug_flag.VALUE:
    import MPyDATA.utils.fake_numba as numba
else:
    import numba


def make_flux():
    @numba.njit(**jit_flags)
    def flux(_, arg_1_data, arg_1_i, arg_1_j, arg_2_data_0, arg_2_data_1, arg_2_i, arg_2_j, dd):
        result = (
                np.maximum(0, vector_field_2d.at(arg_2_data_0, arg_2_data_1, arg_2_i, arg_2_j, dd, +.5, 0)) *
                scalar_field_2d.at(arg_1_data, arg_1_i, arg_1_j, dd, 0, 0) +
                np.minimum(0, vector_field_2d.at(arg_2_data_0, arg_2_data_1, arg_2_i, arg_2_j, dd, +.5, 0)) *
                scalar_field_2d.at(arg_1_data, arg_1_i, arg_1_j, dd, 1, 0)
        )
        return result
    return flux

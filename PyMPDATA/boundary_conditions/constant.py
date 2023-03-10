""" boundary condition filling halos with a constant value """
# pylint: disable=too-many-arguments
from functools import lru_cache

import numba

from PyMPDATA.impl.traversals_common import make_fill_halos_loop


class Constant:
    """class which instances are to be passed in boundary_conditions tuple to the
    `PyMPDATA.scalar_field.ScalarField` and
    `PyMPDATA.vector_field.VectorField` __init__ methods"""

    def __init__(self, value):
        self.value = value

    def make_scalar(self, ats, _, set_value, halo, dtype, jit_flags, dimension_index):
        """returns (lru-cached) Numba-compiled scalar halo-filling callable"""
        return _make_scalar_constant(self.value, ats, set_value, halo, dtype, jit_flags)

    def make_vector(self, ats, _, set_value, halo, dtype, jit_flags, dimension_index):
        """returns (lru-cached) Numba-compiled vector halo-filling callable"""
        return _make_scalar_constant(self.value, ats, set_value, halo, dtype, jit_flags)


@lru_cache()
def _make_scalar_constant(value, _, set_value, __, ___, jit_flags):
    @numba.njit(**jit_flags)
    def fill_halos(_1, _2, _3):
        return value

    return make_fill_halos_loop(jit_flags, set_value, fill_halos)

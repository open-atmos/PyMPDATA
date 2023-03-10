""" periodic/cyclic boundary condition logic """
from functools import lru_cache

import numba

from PyMPDATA.impl.enumerations import SIGN_LEFT, SIGN_RIGHT
from PyMPDATA.impl.traversals_common import make_fill_halos_loop, make_fill_halos_loop_vector


class Periodic:
    """class which instances are to be passed in boundary_conditions tuple to the
    `PyMPDATA.scalar_field.ScalarField` and
    `PyMPDATA.vector_field.VectorField` __init__ methods"""

    def __init__(self):
        assert SIGN_RIGHT == -1
        assert SIGN_LEFT == +1

    @staticmethod
    def make_scalar(ats, _, set_value, __, ___, jit_flags, dimension_index):
        """returns (lru-cached) Numba-compiled scalar halo-filling callable"""
        return _make_scalar_periodic(ats, set_value, jit_flags)

    @staticmethod
    def make_vector(_, atv, set_value, __, ___, jit_flags, dimension_index):
        """returns (lru-cached) Numba-compiled vector halo-filling callable"""
        return _make_vector_periodic(atv, set_value, jit_flags, dimension_index)


@lru_cache()
def _make_scalar_periodic(ats, set_value, jit_flags):
    @numba.njit(**jit_flags)
    def fill_halos(focus_psi, span, sign):
        return ats(*focus_psi, sign * span)

    return make_fill_halos_loop(jit_flags, set_value, fill_halos)


@lru_cache()
def _make_vector_periodic(atv, set_value, jit_flags, dimension_index):
    @numba.njit(**jit_flags)
    def fill_halos_parallel(focus_psi, span, sign):
        if sign == SIGN_LEFT:
            return atv(*focus_psi, sign * (span - 0.5))
        if sign == SIGN_RIGHT:
            return atv(*focus_psi, sign * (span - 1.5))

    @numba.njit(**jit_flags)
    def fill_halos_normal(focus_psi, span, sign):
        # TODO: same as scalar
        pass

    return make_fill_halos_loop_vector(
        jit_flags,
        set_value,
        fill_halos_parallel,
        dimension_index
    )

""" periodic/cyclic boundary condition logic """
from functools import lru_cache

import numba

from PyMPDATA.impl.enumerations import SIGN_LEFT, SIGN_RIGHT
from PyMPDATA.impl.traversals_common import make_fill_halos_loop


class Periodic:
    """class which instances are to be passed in boundary_conditions tuple to the
    `PyMPDATA.scalar_field.ScalarField` and
    `PyMPDATA.vector_field.VectorField` __init__ methods"""

    def __init__(self):
        assert SIGN_RIGHT == -1
        assert SIGN_LEFT == +1

    @staticmethod
    def make_scalar(ats, set_value, _, __, jit_flags):
        """returns (lru-cached) Numba-compiled scalar halo-filling callable"""
        return _make_scalar_periodic(ats, set_value, jit_flags)

    @staticmethod
    def make_vector(ats, set_value, _, __, jit_flags):
        """returns (lru-cached) Numba-compiled vector halo-filling callable"""
        return _make_vector_periodic(ats, set_value, jit_flags)


@lru_cache()
def _make_scalar_periodic(ats, set_value, jit_flags):
    @numba.njit(**jit_flags)
    def fill_halos(focus_psi, span, sign):
        return ats(*focus_psi, sign * span)

    return make_fill_halos_loop(jit_flags, set_value, fill_halos)


@lru_cache()
def _make_vector_periodic(ats, set_value, jit_flags):
    @numba.njit(**jit_flags)
    def fill_halos(focus_psi, span, sign):
        return ats(*focus_psi, sign * span)

    return make_fill_halos_loop(jit_flags, set_value, fill_halos)

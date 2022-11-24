""" periodic/cyclic boundary condition logic """
from functools import lru_cache

import numba

from PyMPDATA.impl.enumerations import SIGN_LEFT, SIGN_RIGHT


class Periodic:
    """class which instances are to be passed in boundary_conditions tuple to the
    `PyMPDATA.scalar_field.ScalarField` and
    `PyMPDATA.vector_field.VectorField` __init__ methods"""

    def __init__(self):
        assert SIGN_RIGHT == -1
        assert SIGN_LEFT == +1

    @staticmethod
    def make_scalar(ats, _, __, jit_flags):
        """returns (lru-cached) Numba-compiled scalar halo-filling callable"""
        return _make_scalar_periodic(ats, jit_flags)

    @staticmethod
    def make_vector(ats, _, __, jit_flags):
        """returns (lru-cached) Numba-compiled vector halo-filling callable"""
        return _make_vector_periodic(ats, jit_flags)


@lru_cache()
def _make_scalar_periodic(ats, jit_flags):
    @numba.njit(**jit_flags)
    def fill_halos(psi, span, sign):
        return ats(*psi, sign * span)

    return fill_halos


@lru_cache()
def _make_vector_periodic(ats, jit_flags):
    @numba.njit(**jit_flags)
    def fill_halos(psi, span, sign):
        return ats(*psi, sign * span)

    return fill_halos

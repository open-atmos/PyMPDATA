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
        print("BQQ")
        return ats(*focus_psi, sign * span)

    # <TODO>
    @numba.njit(**jit_flags)
    def fill_halos_loop(i_rng, j_rng, k_rng, psi, span, sign):
        for i in i_rng:
            for j in j_rng:
                for k in k_rng:
                    focus = (i, j, k)
                    set_value(psi, *focus, fill_halos((focus, psi), span, sign))

    # </TODO>

    return fill_halos_loop


@lru_cache()
def _make_vector_periodic(ats, set_value, jit_flags):
    @numba.njit(**jit_flags)
    def fill_halos(focus_psi, span, sign):
        return ats(*focus_psi, sign * span)

    @numba.njit(**jit_flags)
    def fill_halos_loop(i_rng, j_rng, k_rng, psi, span, sign):
        for i in i_rng:
            for j in j_rng:
                for k in k_rng:
                    focus = (i, j, k)
                    set_value(psi, *focus, fill_halos((focus, psi), span, sign))

    return fill_halos_loop

""" periodic/cyclic boundary condition logic """
from functools import lru_cache

import numba

from PyMPDATA.impl.enumerations import SIGN_LEFT, SIGN_RIGHT
from PyMPDATA.impl.traversals_common import (
    make_fill_halos_loop,
    make_fill_halos_loop_vector,
)


class Periodic:
    """class which instances are to be passed in boundary_conditions tuple to the
    `PyMPDATA.scalar_field.ScalarField` and
    `PyMPDATA.vector_field.VectorField` __init__ methods"""

    def __init__(self):
        assert SIGN_RIGHT == -1
        assert SIGN_LEFT == +1

    @staticmethod
    def make_scalar(indexers, __, ___, jit_flags, dimension_index):
        """returns (lru-cached) Numba-compiled scalar halo-filling callable"""
        return _make_scalar_periodic(
            indexers.ats[dimension_index], indexers.set, jit_flags
        )

    @staticmethod
    def make_vector(indexers, __, ___, jit_flags, dimension_index):
        """returns (lru-cached) Numba-compiled vector halo-filling callable"""
        return _make_vector_periodic(
            indexers.atv, indexers.set, jit_flags, dimension_index, indexers.n_dims
        )


@lru_cache()
def _make_scalar_periodic(ats, set_value, jit_flags):
    @numba.njit(**jit_flags)
    def fill_halos_scalar(focus_psi, span, sign):
        return ats(*focus_psi, sign * span)

    return make_fill_halos_loop(jit_flags, set_value, fill_halos_scalar)


@lru_cache()
def _make_vector_periodic(atv, set_value, jit_flags, dimension_index, n_dims):
    @numba.njit(**jit_flags)
    def fill_halos_parallel(focus_psi, span, sign):
        return atv[dimension_index](
            *focus_psi, sign * (span - (0.5 if sign == SIGN_LEFT else 1.5))
        )

    @numba.njit(**jit_flags)
    def fill_halos_normal(focus_psi, span, sign, dim):
        # pylint: disable=too-many-return-statements
        if n_dims == 3:
            if dim == 0:
                if dimension_index == 1:
                    return atv[0](*focus_psi, 0.5, sign * span, 0)
                if dimension_index == 2:
                    return atv[0](*focus_psi, 0.5, 0, sign * span)
            if dim == 1:
                if dimension_index == 0:
                    return atv[0](*focus_psi, sign * span, 0.5, 0)
                if dimension_index == 2:
                    return atv[0](*focus_psi, 0, 0.5, sign * span)
            if dim == -1:
                if dimension_index == 0:
                    return atv[0](*focus_psi, sign * span, 0, 0.5)
                if dimension_index == 1:
                    return atv[0](*focus_psi, 0, sign * span, 0.5)
        return atv[dimension_index](*focus_psi, sign * span, 0.5)

    return make_fill_halos_loop_vector(
        jit_flags, set_value, fill_halos_parallel, fill_halos_normal, dimension_index
    )

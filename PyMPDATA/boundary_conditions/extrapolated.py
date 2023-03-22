""" boundary condition extrapolating values from the edge to the halo for scalars
and returning edge-of-the-domain value for vectors (with all negative scalar values
set to zero) """
# pylint: disable=too-many-arguments
from functools import lru_cache

import numba
import numpy as np

from PyMPDATA.impl.enumerations import (
    ARG_FOCUS,
    INNER,
    META_AND_DATA_DATA,
    META_AND_DATA_META,
    SIGN_LEFT,
)
from PyMPDATA.impl.traversals_common import (
    make_fill_halos_loop,
    make_fill_halos_loop_vector,
)


class Extrapolated:
    """class which instances are to be passed in boundary_conditions tuple to the
    `PyMPDATA.scalar_field.ScalarField` and
    `PyMPDATA.vector_field.VectorField` __init__ methods"""

    def __init__(self, dim=INNER, eps=1e-10):
        self.eps = eps
        self.dim = dim

    def make_scalar(self, indexers, halo, dtype, jit_flags, dimension_index):
        """returns (lru-cached) Numba-compiled scalar halo-filling callable"""
        return _make_scalar_extrapolated(
            self.dim,
            self.eps,
            indexers.ats[dimension_index],
            indexers.set,
            halo,
            dtype,
            jit_flags,
        )

    @staticmethod
    def make_vector(indexers, _, __, jit_flags, dimension_index):
        """returns (lru-cached) Numba-compiled vector halo-filling callable"""
        return _make_vector_extrapolated(
            indexers.atv[dimension_index], indexers.set, jit_flags, dimension_index
        )


@lru_cache()
# pylint: disable=too-many-arguments
def _make_scalar_extrapolated(dim, eps, ats, set_value, halo, dtype, jit_flags):
    @numba.njit(**jit_flags)
    def impl(psi, span, sign):
        if sign == SIGN_LEFT:
            edg = halo - psi[ARG_FOCUS][dim]
            nom = ats(*psi, edg + 1) - ats(*psi, edg)
            den = ats(*psi, edg + 2) - ats(*psi, edg + 1)
            cnst = nom / den if abs(den) > eps else 0
            return max(ats(*psi, 1) - (ats(*psi, 2) - ats(*psi, 1)) * cnst, 0)
        edg = span + halo - 1 - psi[ARG_FOCUS][dim]
        den = ats(*psi, edg - 1) - ats(*psi, edg - 2)
        nom = ats(*psi, edg) - ats(*psi, edg - 1)
        cnst = nom / den if abs(den) > eps else 0
        return max(ats(*psi, -1) + (ats(*psi, -1) - ats(*psi, -2)) * cnst, 0)

    if dtype == complex:

        @numba.njit(**jit_flags)
        def fill_halos_scalar(psi, span, sign):
            return complex(
                impl(
                    (psi[META_AND_DATA_META], psi[META_AND_DATA_DATA].real), span, sign
                ),
                impl(
                    (psi[META_AND_DATA_META], psi[META_AND_DATA_DATA].imag), span, sign
                ),
            )

    else:

        @numba.njit(**jit_flags)
        def fill_halos_scalar(psi, span, sign):
            return impl(psi, span, sign)

    return make_fill_halos_loop(jit_flags, set_value, fill_halos_scalar)


@lru_cache()
def _make_vector_extrapolated(atv, set_value, jit_flags, dimension_index):
    @numba.njit(**jit_flags)
    def fill_halos_parallel(focus_psi, _, sign):
        return atv(*focus_psi, sign + 0.5)

    @numba.njit(**jit_flags)
    def fill_halos_normal(_, __, ___, ____):
        return np.nan

    return make_fill_halos_loop_vector(
        jit_flags, set_value, fill_halos_parallel, fill_halos_normal, dimension_index
    )

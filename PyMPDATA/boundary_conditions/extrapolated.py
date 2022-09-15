""" boundary condition extrapolating values from the edge to the halo """
from functools import lru_cache

import numba

from PyMPDATA.impl.enumerations import (
    ARG_FOCUS,
    INNER,
    META_AND_DATA_DATA,
    META_AND_DATA_META,
    SIGN_LEFT,
)


class Extrapolated:
    """class which instances are to be passed in boundary_conditions tuple to the
    `PyMPDATA.scalar_field.ScalarField` and
    `PyMPDATA.vector_field.VectorField` __init__ methods"""

    def __init__(self, dim=INNER, eps=1e-10):
        self.eps = eps
        self.dim = dim

    def make_scalar(self, ats, halo, dtype, jit_flags):
        """returns (lru-cached) Numba-compiled scalar halo-filling callable"""
        return _make_scalar(self.dim, self.eps, ats, halo, dtype, jit_flags)

    def make_vector(self, ats, halo, dtype, jit_flags):
        """returns (lru-cached) Numba-compiled vector halo-filling callable"""
        return _make_vector(self.dim, ats, halo, dtype, jit_flags)


@lru_cache()
# pylint: disable=too-many-arguments
def _make_scalar(dim, eps, ats, halo, dtype, jit_flags):
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

    return fill_halos_scalar


@lru_cache()
def _make_vector(_, ats, __, ___, jit_flags):
    @numba.njit(**jit_flags)
    def fill_halos(psi, ____, sign):
        return ats(*psi, sign)

    return fill_halos

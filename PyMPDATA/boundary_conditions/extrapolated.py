"""
boundary condition extrapolating values from the edge to the halo
"""
from functools import lru_cache
import numba
from PyMPDATA.impl.enumerations import ARG_FOCUS, INNER, SIGN_LEFT
from PyMPDATA.impl.enumerations import META_AND_DATA_META, META_AND_DATA_DATA


@lru_cache()
def _make_scalar(dim, eps, ats, halo, dtype, jit_flags):
    @numba.njit(**jit_flags)
    def impl(psi, n, sign):
        if sign == SIGN_LEFT:
            edg = halo - psi[ARG_FOCUS][dim]
            nom = ats(*psi, edg + 1) - ats(*psi, edg)
            den = ats(*psi, edg + 2) - ats(*psi, edg + 1)
            cnst = nom / den if abs(den) > eps else 0
            return max(ats(*psi, 1) - (ats(*psi, 2) - ats(*psi, 1)) * cnst, 0)
        edg = n + halo - 1 - psi[ARG_FOCUS][dim]
        den = ats(*psi, edg - 1) - ats(*psi, edg - 2)
        nom = ats(*psi, edg) - ats(*psi, edg - 1)
        cnst = nom/den if abs(den) > eps else 0
        return max(ats(*psi, -1) + (ats(*psi, -1) - ats(*psi, -2)) * cnst, 0)

    if dtype == complex:
        @numba.njit(**jit_flags)
        def fill_halos_scalar(psi, n, sign):
            return complex(
                impl((psi[META_AND_DATA_META], psi[META_AND_DATA_DATA].real), n, sign),
                impl((psi[META_AND_DATA_META], psi[META_AND_DATA_DATA].imag), n, sign)
            )
    else:
        @numba.njit(**jit_flags)
        def fill_halos_scalar(psi, n, sign):
            return impl(psi, n, sign)

    return fill_halos_scalar


@lru_cache()
def _make_vector(dim, ats, halo, dtype, jit_flags):
    @numba.njit(**jit_flags)
    def fill_halos(psi, _, sign):
        return ats(*psi, sign)
    return fill_halos


class Extrapolated:
    def __init__(self, dim=INNER, eps=1e-10):
        self._eps = eps
        self.dim = dim

    def make_scalar(self, ats, halo, dtype, jit_flags):
        return _make_scalar(self.dim, self._eps, ats, halo, dtype, jit_flags)

    def make_vector(self, ats, halo, dtype, jit_flags):
        return _make_vector(self.dim, ats, halo, dtype, jit_flags)

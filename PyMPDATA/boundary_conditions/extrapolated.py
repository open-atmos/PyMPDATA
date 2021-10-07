from functools import lru_cache
import numba
from PyMPDATA.impl.enumerations import ARG_FOCUS, INNER, SIGN_LEFT
from PyMPDATA.impl.enumerations import META_AND_DATA_META, META_AND_DATA_DATA


@lru_cache()
def _make_scalar(dim, eps, at, halo, dtype):
    @numba.njit()
    def impl(psi, n, sign):
        if sign == SIGN_LEFT:
            edg = halo - psi[ARG_FOCUS][dim]
            nom = at(*psi, edg + 1) - at(*psi, edg)
            den = at(*psi, edg + 2) - at(*psi, edg + 1)
            cnst = nom / den if abs(den) > eps else 0
            return max(at(*psi, 1) - (at(*psi, 2) - at(*psi, 1)) * cnst, 0)
        edg = n + halo - 1 - psi[ARG_FOCUS][dim]
        den = at(*psi, edg - 1) - at(*psi, edg - 2)
        nom = at(*psi, edg) - at(*psi, edg - 1)
        cnst = nom/den if abs(den) > eps else 0
        return max(at(*psi, -1) + (at(*psi, -1) - at(*psi, -2)) * cnst, 0)

    if dtype == complex:
        @numba.njit()
        def fill_halos_scalar(psi, n, sign):
            return complex(
                impl((psi[META_AND_DATA_META], psi[META_AND_DATA_DATA].real), n, sign),
                impl((psi[META_AND_DATA_META], psi[META_AND_DATA_DATA].imag), n, sign)
            )
    else:
        @numba.njit()
        def fill_halos_scalar(psi, n, sign):
            return impl(psi, n, sign)

    return fill_halos_scalar


@lru_cache()
def _make_vector(dim, at, dtype):
    @numba.njit()
    def fill_halos(psi, _, sign):
        return at(*psi, sign)
    return fill_halos


class Extrapolated:
    def __init__(self, dim=INNER, eps=1e-10):
        self._eps = eps
        self.dim = dim

    def make_scalar(self, at, halo, dtype):
        return _make_scalar(self.dim, self._eps, at, halo, dtype)

    def make_vector(self, at, dtype):
        return _make_vector(self.dim, at, dtype)

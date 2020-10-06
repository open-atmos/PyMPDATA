import numba
from functools import lru_cache

# TODO: 1D only
X = -1


@lru_cache()
def _make_scalar(eps, at, halo):
    @numba.njit()
    def fill_halos_scalar(psi, n, sign):
        if sign > 0:  # left
            edg = halo - psi[0][0]
            nom = at(*psi, edg + 1, X) - at(*psi, edg, X)
            den = at(*psi, edg + 2, X) - at(*psi, edg + 1, X)
            cnst = nom / den if abs(den) > eps else 0
            return max(at(*psi, 1, X) - (at(*psi, 2, X) - at(*psi, 1, X)) * cnst, 0)
        else:  # right
            edg = n + halo - 1 - psi[0][0]
            den = at(*psi, edg - 1, X) - at(*psi, edg - 2, X)
            nom = at(*psi, edg, X) - at(*psi, edg - 1, X)
            cnst = nom / den if abs(den) > eps else 0
            return max(at(*psi, - 1, X) + (at(*psi, -1, X) - at(*psi, -2, X)) * cnst, 0)
    return fill_halos_scalar


@lru_cache()
def _make_vector(at):
    @numba.njit()
    def fill_halos(psi, _, sign):
        return at(*psi, sign, 0)
    return fill_halos


class ExtrapolatedBoundaryCondition:
    def __init__(self, eps=1e-10):
        self._eps = eps

    def make_scalar(self, at, halo):
        return _make_scalar(self._eps, at, halo)

    def make_vector(self, at):
        return _make_vector(at)

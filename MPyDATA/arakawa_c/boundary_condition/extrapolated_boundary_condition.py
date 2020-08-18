import numba
from ..enumerations import ARG_FOCUS, INNER

# TODO: 1D only
X = -1


class ExtrapolatedBoundaryCondition:
    def __init__(self, eps=1e-10):
        self._eps = eps

    def make_scalar(self, at, halo):
        eps = self._eps

        @numba.njit()
        def fill_halos_scalar(psi, n, sign):
            if sign > 0:  # left
                edg = halo - psi[ARG_FOCUS][INNER]
                nom = at(*psi, edg + 1, X) - at(*psi, edg, X)
                den = at(*psi, edg + 2, X) - at(*psi, edg + 1, X)
                cnst = nom / den if abs(den) > eps else 0
                return max(at(*psi, 1, X) - (at(*psi, 2, X) - at(*psi, 1, X)) * cnst, 0)
            else:  # right
                edg = n + halo - 1 - psi[ARG_FOCUS][INNER]
                den = at(*psi, edg - 1, X) - at(*psi, edg - 2, X)
                nom = at(*psi, edg, X) - at(*psi, edg - 1, X)
                cnst = nom/den if abs(den) > eps else 0
                return max(at(*psi, - 1, X) + (at(*psi, -1, X) - at(*psi, -2, X)) * cnst, 0)
        return fill_halos_scalar

    def make_vector(self, at):
        @numba.njit()
        def fill_halos(psi, _, sign):
            return at(*psi, sign, 0)
        return fill_halos

import numba
from MPyDATA.jit_flags import jit_flags

# TODO: allow passing from caller scope
eps = 1e-10

# TODO: 1D only
X=-1


class Extrapolated:
    @staticmethod
    def make_scalar(at, halo):
        @numba.njit(**jit_flags)
        def fill_halos_scalar(psi, n, sign):
            if sign > 0:  # left
                edg = halo - psi[0][0]
                nom = at(*psi, edg + 1, X) - at(*psi, edg, X)
                den = at(*psi, edg + 2, X) - at(*psi, edg + 1, X)
                cnst = nom / den if abs(den) > eps else 0
                # print("L: ", n, edg, psi[0][0], cnst)
                return max(at(*psi, 1, X) - (at(*psi, 2, X) - at(*psi, 1, X)) * cnst, 0)
            else:  # right
                edg = n + halo - 1 - psi[0][0]
                den = at(*psi, edg - 1, X) - at(*psi, edg - 2, X)
                nom = at(*psi, edg, X) - at(*psi, edg - 1, X)
                cnst = nom/den if abs(den) > eps else 0
                # print("R: ", n, edg, psi[0][0], cnst)
                return max(at(*psi, - 1, X) + (at(*psi, -1, X) - at(*psi, -2, X)) * cnst, 0)
        return fill_halos_scalar

    @staticmethod
    def make_vector(at):
        @numba.njit(**jit_flags)
        def fill_halos_vector(psi, n, sign):
            return 0 # TODO
        return fill_halos_vector
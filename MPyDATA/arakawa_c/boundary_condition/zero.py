import numba
from ...formulae.jit_flags import jit_flags


class Extrapolated:
    @staticmethod
    def make(_, __):
        @numba.njit(**jit_flags)
        def fill_halos(psi, n, sign):
            return 0
        return fill_halos, fill_halos
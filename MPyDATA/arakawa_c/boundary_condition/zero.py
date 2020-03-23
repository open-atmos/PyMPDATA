import numba
from ...formulae.jit_flags import jit_flags


class Zero:
    @staticmethod
    def make_scalar(_, __):
        @numba.njit(**jit_flags)
        def fill_halos(psi, n, sign):
            return 0
        return fill_halos

    @staticmethod
    def make_vector(_):
        return Zero.make_scalar(_, 0)

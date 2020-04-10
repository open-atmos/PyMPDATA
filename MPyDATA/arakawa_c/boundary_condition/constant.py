import numba
from MPyDATA.jit_flags import jit_flags


class Constant:
    def __init__(self, value):
        self._value = value

    def make_scalar(self, _at, _halo):
        value = self._value

        @numba.njit(**jit_flags)
        def fill_halos(_, __, ___):
            return value
        return fill_halos

    def make_vector(self, at):
        return Constant(self._value).make_scalar(at, 0)

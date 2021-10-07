from functools import lru_cache
import numba


@lru_cache()
def _make_scalar(value, at, halo, jit_flags):
    @numba.njit(**jit_flags, inline='always')
    def fill_halos(_, __, ___):
        return value

    return fill_halos


class Constant:
    def __init__(self, value):
        self.value = value

    def make_scalar(self, at, halo, dtype, jit_flags):
        return _make_scalar(self.value, at, halo, jit_flags)

    def make_vector(self, at, dtype, jit_flags):
        return Constant(self.value).make_scalar(at, 0, dtype, jit_flags)

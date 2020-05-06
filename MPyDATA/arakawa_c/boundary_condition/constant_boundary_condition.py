import numba


class ConstantBoundaryCondition:
    def __init__(self, value):
        self._value = value

    def make_scalar(self, _at, _halo):
        value = self._value

        @numba.njit()
        def fill_halos(_, __, ___):
            return value
        return fill_halos

    def make_vector(self, at):
        return ConstantBoundaryCondition(self._value).make_scalar(at, 0)

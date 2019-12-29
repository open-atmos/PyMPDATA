"""
Created at 07.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from .field import Field
from .vector_fields_utils import _is_integral, _is_fractional
import numpy as np

from ...utils import debug_flag

if debug_flag.VALUE:
    import MPyDATA.utils.fake_numba as numba
else:
    import numba


def make_vector_field_2d(arg_data, arg_halo: int):
    assert arg_data[0].shape[0] == arg_data[1].shape[0] + 1
    assert arg_data[0].shape[1] == arg_data[1].shape[1] - 1
    assert arg_halo >= 0

    halo = int(arg_halo)
    shape = (arg_data[1].shape[0], arg_data[0].shape[1])

    @numba.jitclass([
        ('_data_0', numba.float64[:, :]),
        ('_data_1', numba.float64[:, :]),
        ('_i', numba.int64),
        ('_j', numba.int64),
        ('axis', numba.int64)
    ])
    class VectorField2D:
        def __init__(self, data_0: np.ndarray, data_1: np.ndarray):
            self.axis = 0
            self._data_0 = np.full((
                data_0.shape[0] + 2 * (halo - 1),
                data_0.shape[1] + 2 * halo
            ), np.nan, dtype=np.float64)
            self._data_1 = np.full((
                data_1.shape[0] + 2 * halo,
                data_1.shape[1] + 2 * (halo - 1)
            ), np.nan, dtype=np.float64)
            self._i = 0
            self._j = 0
            self.get_component(0)[:, :] = data_0[:, :]
            self.get_component(1)[:, :] = data_1[:, :]

        def clone(self):
            return VectorField2D(
                self.get_component(0).copy(),
                self.get_component(1).copy()
            )

        def data(self, i) -> np.ndarray:
            if i == 0:
                return self._data_0
            elif i == 1:
                return self._data_1
            else:
                raise ValueError()

        @property
        def dimension(self) -> int:
            return 2

        def focus(self, i: int, j: int):
            self._i = i + halo - 1
            self._j = j + halo - 1

        def set_axis(self, axis: int):
            self.axis = axis

        def at(self, arg1: [int, float], arg2: [int, float]):
            d, idx1, idx2 = self.__idx(arg1, arg2)
            return self.data(d)[idx1, idx2]

        def __idx(self, arg1: [int, float], arg2: [int, float]):
            if self.axis == 1:
                arg1, arg2 = arg2, arg1

            if _is_integral(arg1) and _is_fractional(arg2):
                d = 1
                idx1 = arg1 + 1
                idx2 = int(arg2 + .5)
                assert idx2 == arg2 + .5
            elif _is_integral(arg2) and _is_fractional(arg1):
                d = 0
                idx1 = int(arg1 + .5)
                idx2 = arg2 + 1
                assert idx1 == arg1 + .5
            else:
                raise ValueError()

            # TODO: rely on tests
            assert self._i + idx1 >= 0
            assert self._j + idx2 >= 0

            return d, int(self._i + idx1), int(self._j + idx2)

        def get_component(self, i: int) -> np.ndarray:
            domain = (
                slice(
                    halo - 1,
                    halo - 1 + shape[0] + 1
                ),
                slice(
                    halo,
                    halo + shape[1]
                )
            ) if i == 0 else (
                slice(
                    halo,
                    halo + shape[0]
                ),
                slice(
                    halo - 1,
                    halo - 1 + shape[1] + 1
                )
            )
            return self.data(i)[domain]

        def apply_2arg(self, function: callable, arg_1: Field.Impl, arg_2: Field.Impl, ext: int):
            for i in range(-1-ext, shape[0]+ext):
                for j in range(-1-ext, shape[1]+ext):
                    self.focus(i, j)
                    arg_1.focus(i, j)
                    arg_2.focus(i, j)

                    for dd in range(2):
                        if (i == -1 and dd == 1) or (j == -1 and dd == 0):
                            continue

                        self.set_axis(dd)
                        d, idx_i, idx_j = self.__idx(+.5, 0)
                        self.data(d)[idx_i, idx_j] = 0
                        arg_1.set_axis(dd)
                        arg_2.set_axis(dd)

                        self.data(d)[idx_i, idx_j] += function(arg_1, arg_2)

        # TODO: replace Nones with actual numbers (which are constant)
        def left_halo(self, a: int, c: int):
            if c == 0:
                if a == 0: return slice(0, halo - 1), slice(None, None)
                if a == 1: return slice(None, None), slice(0, halo)
            if c == 1:
                if a == 0: return slice(0, halo), slice(None, None)
                if a == 1: return slice(None, None), slice(0, halo - 1)
            raise ValueError()

        def left_edge(self, a: int, c: int):
            if c == 0:
                if a == 0: return slice(halo - 1,2 * (halo - 1)), slice(None, None)
                if a == 1: return slice(None, None), slice(halo, 2 * halo)
            if c == 1:
                if a == 0: return slice(halo, 2 * halo), slice(None, None)
                if a == 1: return slice(None, None), slice(halo - 1, 2 * (halo - 1))
            raise ValueError()

        def right_halo(self, a: int, c: int):
            if c == 0:
                if a == 0: return slice(-(halo - 1), None), slice(None, None)
                if a == 1: return slice(None, None), slice(-halo, None)
            if c == 1:
                if a == 0: return slice(-halo, None), slice(None, None)
                if a == 1: return slice(None, None), slice(-(halo - 1), None)
            raise ValueError()

        def right_edge(self, a: int, c: int):
            if c == 0:
                if a == 0: return slice(-2 * (halo - 1), -(halo - 1)), slice(None, None)
                if a == 1: return slice(None, None), slice(-2 * halo, -halo)
            if c == 1:
                if a == 0: return slice(-2 * halo, -halo), slice(None, None)
                if a == 1: return slice(None, None), slice(-2 * (halo - 1), -(halo - 1))
            raise ValueError()

    return VectorField2D(data_0=arg_data[0], data_1=arg_data[1])

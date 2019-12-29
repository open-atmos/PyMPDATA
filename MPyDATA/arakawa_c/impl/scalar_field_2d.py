"""
Created at 07.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from .field import Field
import numpy as np

from ...utils import debug_flag

if debug_flag.VALUE:
    import MPyDATA.utils.fake_numba as numba
else:
    import numba


def make_scalar_field_2d(arg_data: np.ndarray, arg_halo: int):
    halo = int(arg_halo)
    shape = (arg_data.shape[0] + 2 * halo, arg_data.shape[1] + 2 * halo)

    @numba.jitclass([
        ('_data', numba.float64[:, :]),
        ('_i', numba.int64),
        ('_j', numba.int64),
        ('axis', numba.int64)
    ])
    class ScalarField2D:
        def __init__(self, data: np.ndarray):
            self.axis = 0
            self._data = np.zeros((shape[0], shape[1]), dtype=np.float64)
            self._i = 0
            self._j = 0
            self.get()[:, :] = data[:, :]

        def focus(self, i: int, j: int):
            self._i = i + halo
            self._j = j + halo

        def set_axis(self, axis):
            self.axis = axis

        def at(self, arg1: int, arg2: int):
            if self.axis == 1:
                return self._data[self._i + arg2, self._j + arg1]
            else:
                return self._data[self._i + arg1, self._j + arg2]

        def apply_2arg(self, function: callable, arg1: Field.Impl, arg2: Field.Impl, ext: int):
            for i in range(-ext, shape[0] - 2 * halo + ext):
                for j in range(-ext, shape[1] - 2 * halo + ext):
                    self.focus(i, j)
                    arg1.focus(i, j)
                    arg2.focus(i, j)

                    self._data[self._i, self._j] = 0
                    for dim in range(2):
                        self.set_axis(dim)
                        arg1.set_axis(dim)
                        arg2.set_axis(dim)

                        self._data[self._i, self._j] += function(arg1, arg2)

        @property
        def dimension(self) -> int:
            return 2

        def get(self):
            results = self._data[
                halo: self._data.shape[0] - halo,
                halo: self._data.shape[1] - halo
            ]
            return results

        def left_halo(self, d: int):
            if d == 0: return self._data[: halo, :]
            if d == 1: return self._data[:, : halo]

        def right_halo(self, d: int):
            if d == 0: return self._data[-halo:, :]
            if d == 1: return self._data[:, -halo:]

        def left_edge(self, d: int):
            if d == 0: return self._data[halo:2 * halo, :]
            if d == 1: return self._data[:, halo:2 * halo]

        def right_edge(self, d: int):
            if d == 0: return self._data[-2 * halo:-halo, :]
            if d == 1: return self._data[:, -2 * halo:-halo]

    return ScalarField2D(data=arg_data)


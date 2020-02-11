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
        ('data', numba.float64[:, :]),
        ('_i', numba.int64),
        ('_j', numba.int64),
        ('axis', numba.int64)
    ])
    class ScalarField2D:
        def __init__(self, data: np.ndarray):
            self.axis = 0
            self.data = np.zeros((shape[0], shape[1]), dtype=np.float64)
            self._i = 0
            self._j = 0
            self.get()[:, :] = data[:, :]

        def clone(self):
            return ScalarField2D(self.get().copy())

        def focus(self, i: int, j: int):
            self._i = i + halo
            self._j = j + halo

        def set_axis(self, axis):
            self.axis = axis

        def at(self, arg1: int, arg2: int):
            if self.axis == 1:
                return self.data[self._i + arg2, self._j + arg1]
            else:
                return self.data[self._i + arg1, self._j + arg2]

        def sum_1arg(self, function: callable, arg1: Field.Impl, ext: int):
            for i in range(-ext, shape[0] - 2 * halo + ext):
                for j in range(-ext, shape[1] - 2 * halo + ext):
                    self.focus(i, j)
                    arg1.focus(i, j)
                    self.data[self._i, self._j] = 0
                    for dim in range(2):
                        self.set_axis(dim)
                        arg1.set_axis(dim)
                        self.data[self._i, self._j] += function(arg1)

        def min_1arg(self, function: callable, arg1: Field.Impl, ext: int):
            for i in range(-ext, shape[0] - 2 * halo + ext):
                for j in range(-ext, shape[1] - 2 * halo + ext):
                    self.focus(i, j)
                    arg1.focus(i, j)
                    self.data[self._i, self._j] = np.inf
                    for dim in range(2):
                        self.set_axis(dim)
                        arg1.set_axis(dim)
                        self.data[self._i, self._j] = np.minimum(
                            function(arg1),
                            self.data[self._i, self._j]
                        )

        def max_1arg(self, function: callable, arg1: Field.Impl, ext: int):
            for i in range(-ext, shape[0] - 2 * halo + ext):
                for j in range(-ext, shape[1] - 2 * halo + ext):
                    self.focus(i, j)
                    arg1.focus(i, j)
                    self.data[self._i, self._j] = -np.inf
                    for dim in range(2):
                        self.set_axis(dim)
                        arg1.set_axis(dim)
                        self.data[self._i, self._j] = np.maximum(
                            function(arg1),
                            self.data[self._i, self._j]
                        )

        def sum_2arg(self, function: callable, arg1: Field.Impl, arg2: Field.Impl, ext: int):
            for i in range(-ext, shape[0] - 2 * halo + ext):
                for j in range(-ext, shape[1] - 2 * halo + ext):
                    self.focus(i, j)
                    arg1.focus(i, j)
                    arg2.focus(i, j)
                    self.data[self._i, self._j] = 0
                    for dim in range(2):
                        self.set_axis(dim)
                        arg1.set_axis(dim)
                        arg2.set_axis(dim)
                        self.data[self._i, self._j] += function(arg1, arg2)

        def set_2arg(self, function: callable, arg1: Field.Impl, arg2: Field.Impl, ext: int):
            for i in range(-ext, shape[0] - 2 * halo + ext):
                for j in range(-ext, shape[1] - 2 * halo + ext):
                    self.focus(i, j)
                    arg1.focus(i, j)
                    arg2.focus(i, j)
                    self.data[self._i, self._j] = function(arg1, arg2)

        def sum_4arg(self, function: callable, arg1: Field.Impl, arg2: Field.Impl, arg3: Field.Impl, arg4: Field.Impl, ext: int):
            for i in range(-ext, shape[0] - 2 * halo + ext):
                for j in range(-ext, shape[1] - 2 * halo + ext):
                    self.focus(i, j)
                    arg1.focus(i, j)
                    arg2.focus(i, j)
                    arg3.focus(i, j)
                    arg4.focus(i, j)
                    self.data[self._i, self._j] = 0
                    for dim in range(2):
                        self.set_axis(dim)
                        arg1.set_axis(dim)
                        arg2.set_axis(dim)
                        arg3.set_axis(dim)
                        arg4.set_axis(dim)
                        self.data[self._i, self._j] += function(arg1, arg2, arg3, arg4)

        def set_4arg(self, function: callable, arg1: Field.Impl, arg2: Field.Impl, arg3: Field.Impl, arg4: Field.Impl, ext: int):
            for i in range(-ext, shape[0] - 2 * halo + ext):
                for j in range(-ext, shape[1] - 2 * halo + ext):
                    self.focus(i, j)
                    arg1.focus(i, j)
                    arg2.focus(i, j)
                    arg3.focus(i, j)
                    arg4.focus(i, j)
                    self.data[self._i, self._j] = function(arg1, arg2, arg3, arg4)

        @property
        def dimension(self) -> int:
            return 2

        def get(self):
            results = self.data[
                halo: self.data.shape[0] - halo,
                halo: self.data.shape[1] - halo
            ]
            return results

        # TODO: replace Nones with actual numbers (which are constant)
        # TODO: consider saving instances of the slices?
        def left_halo(self, d: int):
            if d == 0: return slice(0, halo), slice(None, None)
            if d == 1: return slice(None, None), slice(0, halo)
            raise ValueError()

        def right_halo(self, d: int):
            if d == 0: return slice(-halo, None), slice(None, None)
            if d == 1: return slice(None, None), slice(-halo, None)
            raise ValueError()

        def left_edge(self, d: int):
            if d == 0: return slice(halo, 2 * halo), slice(None, None)
            if d == 1: return slice(None, None), slice(halo, 2 * halo)
            raise ValueError()

        def right_edge(self, d: int):
            if d == 0: return slice(-2 * halo, -halo), slice(None, None)
            if d == 1: return slice(None, None), slice(-2 * halo, -halo)
            raise ValueError()

    return ScalarField2D(data=arg_data)


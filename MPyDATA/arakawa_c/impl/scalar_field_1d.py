"""
Created at 07.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""


import numpy as np
from MPyDATA.utils import debug_flag

if debug_flag.VALUE:
    import MPyDATA.utils.fake_numba as numba
else:
    import numba


def make_scalar_field_1d(arg_data, arg_halo):
    halo = int(arg_halo)
    shape = int(arg_data.shape[0] + 2 * halo)

    @numba.jitclass([
        ('data', numba.float64[:]),
        ('_i', numba.int64),
        ('axis', numba.int64)
    ])
    class ScalarField1D:
        def __init__(self, data):
            self.axis = 0
            self.data = np.zeros((shape,), dtype=np.float64)
            self.data[halo:shape - halo] = data[:]
            self._i = 0

        def clone(self):
            return ScalarField1D(self.get().copy())

        def focus(self, i):
            self._i = i + halo

        def at(self, item, _):
            return self.data[self._i + item]

        def min_1arg(self, function, arg1, ext): self.set_1arg(function, arg1, ext)
        def max_1arg(self, function, arg1, ext): self.set_1arg(function, arg1, ext)
        def set_1arg(self, function, arg1, ext):
            for i in range(-ext, shape - 2 * halo + ext):
                self.focus(i)
                arg1.focus(i)
                self.data[self._i] = function(arg1)

        def sum_2arg(self, function, arg1, arg2, ext):
            for i in range(-ext, shape - 2 * halo + ext):
                self.focus(i)
                arg1.focus(i)
                arg2.focus(i)
                self.data[self._i] = function(arg1, arg2)

        def sum_4arg(self, function, arg1, arg2, arg3, arg4, ext):
            for i in range(-ext, shape - 2 * halo + ext):
                self.focus(i)
                arg1.focus(i)
                arg2.focus(i)
                arg3.focus(i)
                arg4.focus(i)
                self.data[self._i] = function(arg1, arg2, arg3, arg4)

        @property
        def dimension(self) -> int:
            return 1

        def get(self):
            results = self.data[halo: self.data.shape[0] - halo]
            return results

        def left_halo(self, _):
            return slice(0, halo)

        def left_edge(self, _):
            return slice(halo, 2 * halo)

        def right_halo(self, _):
            return slice(shape - halo, shape)

        def right_edge(self, _):
            return slice(-2 * halo, shape-halo)

    return ScalarField1D(data=arg_data)

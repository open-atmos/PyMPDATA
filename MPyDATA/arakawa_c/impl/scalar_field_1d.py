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
        ('_data', numba.float64[:]),
        ('_i', numba.int64),
        ('axis', numba.int64)
    ])
    class ScalarField1D:
        def __init__(self, data):
            self.axis = 0
            self._data = np.zeros((shape,), dtype=np.float64)
            self._data[halo:shape - halo] = data[:]
            self._i = 0

        def focus(self, i):
            self._i = i + halo

        def at(self, item, _):
            return self._data[self._i + item]

        def apply_1arg(self, function, arg1, ext):
            for i in range(-ext, shape - 2 * halo + ext):
                self.focus(i)
                arg1.focus(i)
                self._data[self._i] = function(arg1)

        def apply_2arg(self, function, arg1, arg2, ext):
            for i in range(-ext, shape - 2 * halo + ext):
                self.focus(i)
                arg1.focus(i)
                arg2.focus(i)
                self._data[self._i] = function(arg1, arg2)

        def apply_4arg(self, function, arg1, arg2, arg3, arg4, ext):
            for i in range(-ext, shape - 2 * halo + ext):
                self.focus(i)
                arg1.focus(i)
                arg2.focus(i)
                arg3.focus(i)
                arg4.focus(i)
                self._data[self._i] = function(arg1, arg2, arg3, arg4)

        @property
        def dimension(self) -> int:
            return 1

        def get(self):
            results = self._data[halo: self._data.shape[0] - halo]
            return results

        def left_halo(self, _):
            return self._data[: halo]

        def left_edge(self, _):
            return self._data[halo:2 * halo]

        def right_halo(self, _):
            return self._data[-halo:]

        def right_edge(self, _):
            return self._data[-2 * halo:-halo]

    return ScalarField1D(data=arg_data)

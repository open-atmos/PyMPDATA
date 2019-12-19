"""
Created at 07.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
from MPyDATA.utils import debug
if debug.DEBUG:
    import MPyDATA.utils.fake_numba as numba
else:
    import numba
from MPyDATA.fields.utils import is_integral


@numba.jitclass([
    ('halo', numba.int64),
    ('shape_0', numba.int64),
    ('data_0', numba.float64[:]),
    ('i', numba.int64),
    ('axis', numba.int64)
])
class VectorField1D:
    def __init__(self, data_0, halo):
        assert halo > 0
        self.axis = 0
        self.halo = halo
        self.shape_0 = data_0.shape[0] - 1
        self.data_0 = np.zeros((data_0.shape[0] + 2 * (halo - 1)), dtype=np.float64)

        shape_with_halo = data_0.shape[0] + 2 * (halo - 1)
        self.data_0[halo - 1:shape_with_halo - (halo - 1)] = data_0[:]

        self.i = 0

    @property
    def dimension(self):
        return 1

    def focus(self, i):
        self.i = i + self.halo - 1

    def at(self, item, _):
        idx = self.__idx(item)
        return self.data_0[idx]

    def __idx(self, item):
        if is_integral(item):
            raise ValueError()
        return self.i + int(item + .5)

    def get_component(self, _):
        return self.data_0[self.halo - 1: self.data_0.shape[0] - self.halo + 1]

    def apply_2arg(self, function, arg_1, arg_2, ext):
        for i in range(-1 - ext, self.shape_0 + ext):
            self.focus(i)
            arg_1.focus(i)
            arg_2.focus(i)

            idx = self.__idx(+.5)
            self.data_0[idx] = function(arg_1, arg_2)

    def fill_halos(self):
        if self.halo == 1:
            return

        hm1 = self.halo - 1
        sp1 = self.shape_0 + 1

        left_halo = slice(0, hm1)
        left_edge = slice(left_halo.stop, 2 * hm1)

        right_edge = slice(sp1, sp1 + hm1)
        right_halo = slice(right_edge.stop, sp1 + 2 * hm1)

        self.data_0[left_halo] = self.data_0[right_edge]
        self.data_0[right_halo] = self.data_0[left_edge]

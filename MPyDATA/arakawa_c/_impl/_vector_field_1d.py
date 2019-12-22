"""
Created at 07.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np
from MPyDATA_tests.utils import debug
if debug.DEBUG:
    import MPyDATA_tests.utils.fake_numba as numba
else:
    import numba
from MPyDATA.arakawa_c._impl._vector_fields_utils import _is_integral


@numba.jitclass([
    ('halo', numba.int64),
    ('shape_0', numba.int64),
    ('_data_0', numba.float64[:]),
    ('_i', numba.int64),
    ('axis', numba.int64),
    ('_halo_valid', numba.boolean)
])
class _VectorField1D:
    def __init__(self, data_0, halo):
        assert halo > 0
        self.axis = 0
        self.halo = halo
        self.shape_0 = data_0.shape[0] - 1
        self._data_0 = np.zeros((data_0.shape[0] + 2 * (halo - 1)), dtype=np.float64)

        shape_with_halo = data_0.shape[0] + 2 * (halo - 1)
        self._data_0[halo - 1:shape_with_halo - (halo - 1)] = data_0[:]

        self._i = 0

        self._halo_valid = False

    @property
    def dimension(self):
        return 1

    def _focus(self, i):
        self._i = i + self.halo - 1

    def at(self, item, _):
        idx = self.__idx(item)
        return self._data_0[idx]

    def __idx(self, item):
        if _is_integral(item):
            raise ValueError()
        return self._i + int(item + .5)

    def get_component(self, _):
        return self._data_0[self.halo - 1: self._data_0.shape[0] - self.halo + 1]

    def _apply_2arg(self, function, arg_1, arg_2, ext):
        for i in range(-1 - ext, self.shape_0 + ext):
            self._focus(i)
            arg_1._focus(i)
            arg_2._focus(i)

            idx = self.__idx(+.5)
            self._data_0[idx] = function(arg_1, arg_2)

    def fill_halos(self):
        if self._halo_valid or self.halo < 2:
            return

        hm1 = self.halo - 1
        sp1 = self.shape_0 + 1

        left_halo = slice(0, hm1)
        left_edge = slice(left_halo.stop, 2 * hm1)

        right_edge = slice(sp1, sp1 + hm1)
        right_halo = slice(right_edge.stop, sp1 + 2 * hm1)

        self._data_0[left_halo] = self._data_0[right_edge]
        self._data_0[right_halo] = self._data_0[left_edge]

        self._halo_valid = True

    def invalidate_halos(self):
        self._halo_valid = False
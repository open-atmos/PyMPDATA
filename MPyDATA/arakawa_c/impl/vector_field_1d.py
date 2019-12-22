"""
Created at 07.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""
from .vector_fields_utils import _is_integral
from .field import Field
import numpy as np

from MPyDATA_tests.utils import debug
if debug.DEBUG:
    import MPyDATA_tests.utils.fake_numba as numba
else:
    import numba


@numba.jitclass([
    ('halo', numba.int64),
    ('shape_0', numba.int64),
    ('_data_0', numba.float64[:]),
    ('_i', numba.int64),
    ('axis', numba.int64),
    ('_halo_valid', numba.boolean)
])
class VectorField1D:
    def __init__(self, data_0: np.ndarray, halo: int):
        assert halo > 0
        self.axis = 0
        self.halo = halo
        self.shape_0 = data_0.shape[0] - 1

        self._data_0 = np.zeros((data_0.shape[0] + 2 * (halo - 1)), dtype=np.float64)
        self._i = 0
        self._halo_valid = False

        self.get_component(0)[:] = data_0[:]

    @property
    def dimension(self):
        return 1

    def focus(self, i):
        self._i = i + self.halo - 1

    def at(self, item: float, _):
        idx = self.__idx(item)
        return self._data_0[idx]

    def __idx(self, item: float):
        if _is_integral(item):
            raise ValueError()
        return self._i + int(item + .5)

    def get_component(self, _):
        return self._data_0[self.halo - 1: self._data_0.shape[0] - self.halo + 1]

    def apply_2arg(self, function: callable, arg_1: Field.Impl, arg_2: Field.Impl, ext: int):
        for i in range(-1 - ext, self.shape_0 + ext):
            self.focus(i)
            arg_1.focus(i)
            arg_2.focus(i)

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

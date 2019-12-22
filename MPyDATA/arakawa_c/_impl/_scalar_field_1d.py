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


@numba.jitclass([
    ('halo', numba.int64),
    ('shape', numba.int64[:]),
    ('_data', numba.float64[:]),
    ('_i', numba.int64),
    ('axis', numba.int64),
    ('_halo_valid', numba.boolean)
])
class _ScalarField1D:
    def __init__(self, data, halo):
        self.axis = 0
        self.halo = halo
        self.shape = np.array([data.shape[0] + 2 * halo])

        self._data = np.zeros(self.shape[0], dtype=np.float64)
        self._data[halo:self.shape[0] - halo] = data[:]

        self._i = 0
        self._halo_valid = False

    def _focus(self, i):
        self._i = i + self.halo

    def at(self, item, _):
        return self._data[self._i + item]

    def _apply_1arg(self, function, arg1, ext):
        for i in range(-ext, self.shape[0] - 2 * self.halo + ext):
            self._focus(i)
            arg1._focus(i)
            self._data[self._i] = function(arg1)

    def _apply_2arg(self, function, arg1, arg2, ext):
        for i in range(-ext, self.shape[0] - 2 * self.halo + ext):
            self._focus(i)
            arg1._focus(i)
            arg2._focus(i)
            self._data[self._i] = function(arg1, arg2)

    def apply_4arg(self, function, arg1, arg2, arg3, arg4, ext):
        for i in range(-ext, self.shape[0] - 2 * self.halo + ext):
            self._focus(i)
            arg1._focus(i)
            arg2._focus(i)
            arg3._focus(i)
            arg4._focus(i)
            self._data[self._i] = function(arg1, arg2, arg3, arg4)

    @property
    def dimension(self) -> int:
        return 1

    def get(self):
        results = self._data[self.halo: self._data.shape[0] - self.halo]
        return results

    def fill_halos(self):
        if self._halo_valid or self.halo == 0:
            return

        # TODO: hardcoded periodic boundary
        self._data[: self.halo] = self._data[-2 * self.halo:-self.halo]
        self._data[-self.halo:] = self._data[self.halo:2 * self.halo]

        self._halo_valid = True

    def invalidate_halos(self):
        self._halo_valid = False
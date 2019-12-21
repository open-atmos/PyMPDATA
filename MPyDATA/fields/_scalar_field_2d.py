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
    ('_data', numba.float64[:, :]),
    ('_i', numba.int64),
    ('_j', numba.int64),
    ('axis', numba.int64),
    ('_halo_valid', numba.boolean)
])
class _ScalarField2D:
    def __init__(self, data, halo):
        self.halo = halo
        self.shape = np.zeros(2, dtype=np.int64)
        self.shape[0] = data.shape[0] + 2 * halo
        self.shape[1] = data.shape[1] + 2 * halo

        self._data = np.zeros((self.shape[0], self.shape[1]), dtype=np.float64)
        self._data[halo:self.shape[0] - halo, halo:self.shape[1] - halo] = data[:, :]

        self._i = 0
        self._j = 0
        self.axis = 0

        self._halo_valid = False

    def _focus(self, i, j):
        self._i = i + self.halo
        self._j = j + self.halo

    def _set_axis(self, axis):
        self.axis = axis

    def swap_memory(self, other):
        self._data, other._data = other._data, self._data
        self._halo_valid, other._halo_valid = other._halo_valid, self._halo_valid

    def at(self, arg1, arg2):
        if self.axis == 1:
            return self._data[self._i + arg2, self._j + arg1]
        else:
            return self._data[self._i + arg1, self._j + arg2]

    def _apply_2arg(self, function, arg1, arg2, ext):
        for i in range(-ext, self.shape[0] - 2 * self.halo + ext):
            for j in range(-ext, self.shape[1] - 2 * self.halo + ext):
                self._focus(i, j)
                arg1._focus(i, j)
                arg2._focus(i, j)

                self._data[self._i, self._j] = 0
                for dim in range(2):
                    self._set_axis(dim)
                    arg1._set_axis(dim)
                    arg2._set_axis(dim)

                    self._data[self._i, self._j] += function(arg1, arg2)

    def get(self):
        results = self._data[
            self.halo: self._data.shape[0] - self.halo,
            self.halo: self._data.shape[1] - self.halo
        ]
        return results

    def fill_halos(self):
        if self._halo_valid or self.halo == 0:
            return

        # TODO: use set_axis and loop over dimensions
        # TODO: hardcoded periodic boundary
        self._data[: self.halo, :] = self._data[-2 * self.halo:-self.halo, :]
        self._data[-self.halo:, :] = self._data[self.halo:2 * self.halo, :]

        self._data[:, : self.halo] = self._data[:, -2 * self.halo:-self.halo]
        self._data[:, -self.halo:] = self._data[:, self.halo:2 * self.halo]

        self._halo_valid = True

    def invalidate_halos(self):
        self._halo_valid = False

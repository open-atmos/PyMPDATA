"""
Created at 07.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""

from .field import Field
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
class ScalarField2D:
    def __init__(self, data: np.ndarray, halo: int):
        self.halo = halo
        self.axis = 0
        self.shape = np.zeros(2, dtype=np.int64)
        self.shape[0] = data.shape[0] + 2 * halo
        self.shape[1] = data.shape[1] + 2 * halo

        self._data = np.zeros((self.shape[0], self.shape[1]), dtype=np.float64)
        self._i = 0
        self._j = 0
        self._halo_valid = False

        self.get()[:, :] = data[:, :]

    def focus(self, i: int, j: int):
        self._i = i + self.halo
        self._j = j + self.halo

    def set_axis(self, axis):
        self.axis = axis

    def at(self, arg1: int, arg2: int):
        if self.axis == 1:
            return self._data[self._i + arg2, self._j + arg1]
        else:
            return self._data[self._i + arg1, self._j + arg2]

    def apply_2arg(self, function: callable, arg1: Field.Impl, arg2: Field.Impl, ext: int):
        for i in range(-ext, self.shape[0] - 2 * self.halo + ext):
            for j in range(-ext, self.shape[1] - 2 * self.halo + ext):
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

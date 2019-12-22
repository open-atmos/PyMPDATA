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

from MPyDATA.arakawa_c._impl._vector_fields_utils import _is_integral, _is_fractional


@numba.jitclass([
    ('halo', numba.int64),
    ('shape', numba.int64[:]),
    ('_data_0', numba.float64[:, :]),
    ('_data_1', numba.float64[:, :]),
    ('_i', numba.int64),
    ('_j', numba.int64),
    ('axis', numba.int64),
    ('_halo_valid', numba.boolean)
])
class _VectorField2D:
    def __init__(self, data_0, data_1, halo):
        self.halo = halo
        self.shape = np.zeros(2, dtype=np.int64)
        self.shape[0] = data_1.shape[0]
        self.shape[1] = data_0.shape[1]

        assert data_0.shape[0] == data_1.shape[0] + 1
        assert data_0.shape[1] == data_1.shape[1] - 1

        self._data_0 = np.full((
            data_0.shape[0] + 2 * (halo - 1),
            data_0.shape[1] + 2 * halo
        ), np.nan, dtype=np.float64)
        self._data_1 = np.full((
            data_1.shape[0] + 2 * halo,
            data_1.shape[1] + 2 * (halo - 1)
        ), np.nan, dtype=np.float64)

        self.get_component(0)[:, :] = data_0[:, :]
        self.get_component(1)[:, :] = data_1[:, :]

        self._i = 0
        self._j = 0
        self.axis = 0
        self._halo_valid = False

    def data(self, i) -> np.ndarray:
        if i == 0:
            return self._data_0
        elif i == 1:
            return self._data_1
        else:
            raise ValueError()

    @property
    def dimension(self) -> int:
        return 2

    def _focus(self, i: int, j: int):
        self._i = i + self.halo - 1
        self._j = j + self.halo - 1

    def _set_axis(self, axis: int):
        self.axis = axis

    def at(self, arg1: int, arg2: int):
        d, idx1, idx2 = self.__idx_2d(arg1, arg2)
        return self.data(d)[idx1, idx2]

    def __idx_2d(self, arg1: int, arg2: int):
        if self.axis == 1:
            arg1, arg2 = arg2, arg1

        if _is_integral(arg1) and _is_fractional(arg2):
            d = 1
            idx1 = arg1 + 1
            idx2 = int(arg2 + .5)
            assert idx2 == arg2 + .5
        elif _is_integral(arg2) and _is_fractional(arg1):
            d = 0
            idx1 = int(arg1 + .5)
            idx2 = arg2 + 1
            assert idx1 == arg1 + .5
        else:
            raise ValueError()

        # TODO: rely on tests
        assert self._i + idx1 >= 0
        assert self._j + idx2 >= 0

        return d, int(self._i + idx1), int(self._j + idx2)

    def get_component(self, i: int) -> np.ndarray:
        domain = (
            slice(
                self.halo - 1,
                self.halo - 1 + self.shape[0] + 1
            ),
            slice(
                self.halo,
                self.halo + self.shape[1]
            )
        ) if i == 0 else (
            slice(
                self.halo,
                self.halo + self.shape[0]
            ),
            slice(
                self.halo - 1,
                self.halo - 1 + self.shape[1] + 1
            )
        )
        return self.data(i)[domain]

    def _apply_2arg(self, function: callable, arg_1: int, arg_2: int, ext: int):
        for i in range(-1-ext, self.shape[0]+ext):
            for j in range(-1-ext, self.shape[1]+ext):
                self._focus(i, j)
                arg_1._focus(i, j)
                arg_2._focus(i, j)

                for dd in range(2):
                    if (i == -1 and dd == 1) or (j == -1 and dd == 0):
                        continue

                    self._set_axis(dd)
                    d, idx_i, idx_j = self.__idx_2d(+.5, 0)
                    self.data(d)[idx_i, idx_j] = 0
                    arg_1._set_axis(dd)
                    arg_2._set_axis(dd)

                    self.data(d)[idx_i, idx_j] += function(arg_1, arg_2)

    def fill_halos(self):
        if self._halo_valid or self.halo == 0:
            return

        if self.halo > 1:
            self._data_0[: (self.halo - 1), :] = self._data_0[-2 * (self.halo - 1):-(self.halo - 1), :]
            self._data_0[-(self.halo - 1):, :] = self._data_0[(self.halo - 1):2 * (self.halo - 1), :]
            self._data_1[:, : (self.halo - 1)] = self._data_1[:, -2 * (self.halo - 1):-(self.halo - 1)]
            self._data_1[:, -(self.halo - 1):] = self._data_1[:, (self.halo - 1):2 * (self.halo - 1)]

        self._data_0[:, : self.halo] = self._data_0[:, -2 * self.halo:-self.halo]
        self._data_0[:, -self.halo:] = self._data_0[:, self.halo:2 * self.halo]
        self._data_1[: self.halo, :] = self._data_1[-2 * self.halo:-self.halo, :]
        self._data_1[-self.halo:, :] = self._data_1[self.halo:2 * self.halo, :]


        self._halo_valid = True

    def invalidate_halos(self):
        self._halo_valid = False


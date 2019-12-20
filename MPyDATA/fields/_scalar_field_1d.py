"""
Created at 07.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""


import numpy as np
from MPyDATA_tests.utils import debug

if debug.DEBUG:
    import MPyDATA_tests.utils.fake_numba as numba
    print("fake")
else:
    import numba
    print("numba!")


@numba.jitclass([
    ('halo', numba.int64),
    ('shape', numba.int64[:]),
    ('data', numba.float64[:]),
    ('i', numba.int64),
    ('axis', numba.int64),
    ('halo_valid', numba.boolean)
])
class ScalarField1D:
    def __init__(self, data, halo):
        self.axis = 0
        self.halo = halo
        self.shape = np.array([data.shape[0] + 2 * halo])

        self.data = np.zeros(self.shape[0], dtype=np.float64)
        self.data[halo:self.shape[0] - halo] = data[:]

        self.i = 0
        self.halo_valid = False

    def focus(self, i):
        self.i = i + self.halo

    def swap_memory(self, other):
        self.data, other.data = other.data, self.data

    def at(self, item, _):
        return self.data[self.i + item]

    def apply_1arg(self, function, arg1, ext):
        for i in range(-ext, self.shape[0] - 2 * self.halo + ext):
            self.focus(i)
            arg1.focus(i)
            self.data[self.i] = function(arg1)

    def apply_2arg(self, function, arg1, arg2, ext):
        for i in range(-ext, self.shape[0] - 2 * self.halo + ext):
            self.focus(i)
            arg1.focus(i)
            arg2.focus(i)
            self.data[self.i] = function(arg1, arg2)

    def apply_4arg(self, function, arg1, arg2, arg3, arg4, ext):
        for i in range(-ext, self.shape[0] - 2 * self.halo + ext):
            self.focus(i)
            arg1.focus(i)
            arg2.focus(i)
            arg3.focus(i)
            arg4.focus(i)
            self.data[self.i] = function(arg1, arg2, arg3, arg4)

    def get(self):
        results = self.data[self.halo: self.data.shape[0] - self.halo]
        return results

    def fill_halos(self):
        if self.halo_valid:
            return

        # TODO: hardcoded periodic boundary
        self.data[: self.halo] = self.data[-2*self.halo:-self.halo]
        self.data[-self.halo:] = self.data[self.halo:2 * self.halo]

        self.halo_valid = True

    def invalidate_halos(self):
        self.halo_valid = False
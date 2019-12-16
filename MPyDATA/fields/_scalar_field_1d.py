"""
Created at 07.11.2019

@author: Piotr Bartman
@author: Sylwester Arabas
"""


import numpy as np
import numba


@numba.jitclass([
    ('halo', numba.int64),
    ('shape', numba.int64[:]),
    ('data', numba.float64[:]),
    ('i', numba.int64),
    ('axis', numba.int64)
])
class ScalarField1D:
    def __init__(self, data, halo):
        self.axis = 0
        self.halo = halo
        self.shape = np.array([data.shape[0] + 2 * halo])

        self.data = np.zeros(self.shape[0], dtype=np.float64)
        self.data[halo:self.shape[0] - halo] = data[:]

        self.i = 0

    def focus(self, i):
        self.i = i + self.halo

    def swap_memory(self, other):
        self.data, other.data = other.data, self.data

    def at(self, item, _):
        return self.data[self.i + item]

    def apply(self, function, arg1, arg2):
        for i in range(self.shape[0] - 2 * self.halo):
            self.focus(i)
            arg1.focus(i)
            arg2.focus(i)
            self.data[self.i] = function(arg1, arg2)

    def get(self):
        results = self.data[self.halo: self.data.shape[0] - self.halo]
        return results

    def fill_halos(self):
        # TODO: hardcoded periodic boundary
        self.data[: self.halo] = self.data[-2*self.halo:-self.halo]
        self.data[-self.halo:] = self.data[self.halo:2 * self.halo]

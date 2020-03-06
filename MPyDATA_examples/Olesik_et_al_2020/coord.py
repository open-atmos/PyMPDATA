"""
Created at 22.07.2019

@author: Michael Olesik
@author: Piotr Bartman
@author: Sylwester Arabas
"""


import numpy as np


class x_id:
    def x(self, r):
        return r

    def r(self, x):
        return x

    def dx_dr(self, r):
        return r**0


class x_ln:
    def __init__(self):
        self.r0 = 1

    def x(self, r):
        return np.log(r / self.r0)

    def r(self, x):
        return self.r0 * np.exp(x)

    def dx_dr(self, r):
        return 1/r


class x_p2:
    def x(self, r):
        return r**2

    def r(self, x):
        return np.sqrt(np.where(x < 0, 1e10, x))

    def dx_dr(self, r):
        return 2*r

############################################################################
# TODO: move
# TODO: rename n_n -> number...; n_m -> mass...
class n_n(x_id):
    @staticmethod
    def to_n_n(y, _, __):
        return y

    @staticmethod
    def to_n_s(y, r1, r2):
        return y * (r2 ** 2 + r1 ** 2 + r1*r2) * 4 / 3 * np.pi

    @staticmethod
    def to_n_v(y, r1, r2):
        return y * (r2 + r1) * (r2 ** 2 + r1 ** 2) / 4 * 4 / 3 * np.pi

    @staticmethod
    def from_n_n(n_n, _):
        return 1 * n_n


class n_s(x_p2):
    @staticmethod
    def to_n_s(_n_s, _, __):
        return _n_s

    @staticmethod
    def to_n_n(_n_s, r1, r2):
        return 3 / 4 / np.pi / (r2 **2 + r1 ** 2 + r1*r2) * _n_s

    @staticmethod
    def to_n_v(_n_s, r1, r2):
        return n_n.to_n_v(n_s.to_n_n(_n_s, r1, r2), r1, r2)

    @staticmethod
    def from_n_n(n_n, r):
        return 4 * np.pi * n_n * r**2





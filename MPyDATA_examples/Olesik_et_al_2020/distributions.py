import numpy as np


class n_n:
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


class n_s:
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


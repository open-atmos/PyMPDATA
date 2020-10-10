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

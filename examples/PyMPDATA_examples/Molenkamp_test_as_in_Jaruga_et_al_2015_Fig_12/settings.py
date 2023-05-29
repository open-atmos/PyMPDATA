import numpy as np
import numba
from pystrict import strict


grid = (100, 100)

dt = .1
dx = 1
dy = 1
omega = .1
h = 4.
h0 = 1

r = 15. * dx
x0 = 50 * dx
y0 = 75 * dy
xc = .5 * grid[0] * dx
yc = .5 * grid[1] * dy


@strict
class Settings:
    def __init__(self, n_rotations: int = 6):
        self.n_rotations = n_rotations

    @property
    def dt(self):
        return dt

    @property
    def nt(self):
        return int(628 * self.n_rotations)

    @property
    def size(self):
        return self.xrange[1], self.yrange[1]

    @property
    def xrange(self):
        return 0, grid[0] * dx

    @property
    def yrange(self):
        return 0, grid[1] * dy

    @property
    def grid(self):
        return grid

    @staticmethod
    @numba.njit()
    def pdf(x, y):
        tmp = (x-x0)**2 + (y-y0)**2
        return h0 + np.where(
            # if
            tmp - r**2 <= 0,
            # then
            h - np.sqrt(tmp / (r/h)**2),
            # else
            0.
        )

    @staticmethod
    def stream_function(xX, yY):
        x = xX * grid[0] * dx
        y = yY * grid[1] * dy
        return 1 / 2 * omega * ((x - xc)**2 + (y - yc)**2)

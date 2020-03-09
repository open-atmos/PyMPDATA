import numpy as np
import numba

grid = (502, 401)
# grid = (5,7)

dt = .1
dx = 1
dy = 1
omega = .1
h = 4.
h0 = 1

# TODO: function of grid! (100x100 assumed)
r = .15 * grid[0] * dx
x0 = int(.5 * grid[0]) * dx
y0 = int(.2 * grid[0]) * dy

xc = .5 * grid[0] * dx
yc = .5 * grid[1] * dy


class Setup:
    def __init__(self, n_rotations=6):
        self.n_rotations = n_rotations

    @property
    def dt(self):
        return dt

    @property
    def nt(self):
        return int(300 * self.n_rotations)

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

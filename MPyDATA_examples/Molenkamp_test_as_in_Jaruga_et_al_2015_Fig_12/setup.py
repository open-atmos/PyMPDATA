import numpy as np
import numba

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


class Setup:

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

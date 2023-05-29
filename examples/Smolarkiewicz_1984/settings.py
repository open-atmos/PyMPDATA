from pystrict import strict
import numpy as np


@strict
class Settings:
    def __init__(self, n: int, dt: float):
        self.grid = (n, n, n)
        self.dt = dt
        self.L = 100
        self.dx = self.L / n
        self.dy = self.dx
        self.dz = self.dx
        self.h = 4
        self.r = 15
        d = 25 / np.sqrt(3)
        self.x0 = 50 - d
        self.y0 = 50 + d
        self.z0 = 50 + d

        self.omega = 0.1
        self.xc = 50
        self.yc = 50
        self.zc = 50

    @property
    def advector(self):
        """ constant angular velocity rotational field """

        data = [None, None, None]
        for index, letter in enumerate(('x', 'y', 'z')):
            i, j, k = np.indices((g + (gi == index) for gi, g in enumerate(self.grid)))
            if letter == 'x':
                data[index] = (-((j+.5) * self.dy - self.yc) + ((k+.5) * self.dz - self.zc))\
                              / self.dx
            elif letter == 'y':
                data[index] = (+((i+.5) * self.dx - self.xc) - ((k+.5) * self.dz - self.zc))\
                              / self.dy
            elif letter == 'z':
                data[index] = (-((i+.5) * self.dx - self.xc) + ((j+.5) * self.dy - self.yc))\
                              / self.dz
            data[index] *= self.omega / np.sqrt(3) * self.dt
        return data

    @property
    def advectee(self):
        i, j, k = np.indices(self.grid)
        dist = (
            ((i+.5) * self.dx - self.x0) ** 2 +
            ((j+.5) * self.dy - self.y0) ** 2 +
            ((k+.5) * self.dz - self.z0) ** 2
        )
        return np.where(dist - pow(self.r, 2) <= 0, self.h, 0)

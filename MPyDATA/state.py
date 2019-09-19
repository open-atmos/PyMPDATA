"""
Created at 22.07.2019

@author: Michael Olesik
@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np

from MPyDATA import numerics as nm


class State:
    def __init__(self, n_halo, nr,  r_min, r_max, dt, cdf_r_lambda, coord):
        self.i = slice(0, nr) + n_halo * nm.ONE
        self.ih = self.i % nm.HALF  # cell-border stuff
        self.dx = self.init_dx(nr,  r_min, r_max, coord)
        self.xh = self.init_xh(n_halo, nr,  r_min, r_max, coord)
        self.rh = coord.r(self.xh)
        self.Gh = 1 / coord.dx_dr(self.rh)
        self.GCh = np.full_like(self.Gh, np.nan)
        self.flx = np.full_like(self.Gh, np.nan)
        # cell-centered stuff
        self.x = self.init_x(n_halo, nr)
        self._r = coord.r(self.x)
        self.G = 1 / coord.dx_dr(self._r)
        self.psi = self.init_psi(cdf_r_lambda)

    def init_psi(self, cdf_r_lambda):
        psi = np.full_like(self.G, np.nan)
        psi[self.i] = (
                np.diff(cdf_r_lambda(self.rh[self.ih]))
                /
                np.diff(self.rh[self.ih])
        )
        return psi

    def init_dx(self, nr,  r_min, r_max, coord):
        _, dx = np.linspace(
            coord.x(r_min),
            coord.x(r_max),
            nr + 1,
            retstep=True
        )
        return dx

    def init_xh(self, n_halo, nr,  r_min, r_max, coord):
        xh = np.linspace(
            coord.x(r_min) - (n_halo - 1) * self.dx,
            coord.x(r_max) + (n_halo - 1) * self.dx,
            nr + 1 + 2 * (n_halo - 1)
        )
        return xh

    def init_x(self, n_halo, nr):
        x = np.linspace(
            self.xh[0] - self.dx / 2,
            self.xh[-1] + self.dx / 2,
            nr + 2 * n_halo
        )
        return x

    def Gpsi_sum(self):
        return np.sum(self.G[self.i] * self.psi[self.i])

    @property
    def pdf(self):
        return self.psi[self.i]

    @property
    def r(self):
        return self._r[self.i]

"""
Created at 22.07.2019

@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np

from numerics import numerics as nm


class State:
    @staticmethod
    def magn(q):
        return q.to_base_units().magnitude

    def __init__(self, n_halo, nr,  r_min, r_max, dt, cdf_r_lambda, coord, nm):
        self.nm = nm

        self.i = slice(0, nr) + n_halo * self.nm.one
        self.ih = self.i % self.nm.half  # cell-border stuff
        x_unit = coord.x(r_min).to_base_units().units
        self.dx = self.init_dx(nr,  r_min, r_max, coord)
        self.xh = self.init_xh(n_halo, nr,  r_min, r_max, coord)
        self.rh = coord.r(self.xh * x_unit)
        self.Gh = 1 / self.magn(coord.dx_dr(self.rh))
        self.GCh = np.full_like(self.Gh, np.nan)
        self.flx = np.full_like(self.Gh, np.nan)
        # cell-centered stuff
        self.x = self.init_x(n_halo, nr)
        self._r = coord.r(self.x * x_unit)
        self.G = 1 / self.magn(coord.dx_dr(self._r))
        self.psi = self.init_psi(cdf_r_lambda)

    def init_psi(self, cdf_r_lambda):
        psi = np.full_like(self.G, np.nan)
        psi[self.i] = (
                np.diff(self.magn(cdf_r_lambda(self.rh[self.ih])))
                /
                np.diff(self.magn(self.rh[self.ih]))
        )
        return psi

    def init_dx(self, nr,  r_min, r_max, coord):
        _, dx = np.linspace(
            self.magn(coord.x(r_min)),
            self.magn(coord.x(r_max)),
            nr + 1,
            retstep=True
        )
        return dx

    def init_xh(self, n_halo, nr,  r_min, r_max, coord):
        xh = np.linspace(
            self.magn(coord.x(r_min)) - (n_halo - 1) * self.dx,
            self.magn(coord.x(r_max)) + (n_halo - 1) * self.dx,
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
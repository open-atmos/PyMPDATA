"""
Created at 20.03.2020
"""

import numba


class PolarBoundaryCondition:
    def __init__(self, grid, longitude_idx):
        nlon = grid[longitude_idx]
        assert nlon % 2 == 0

        self.nlon_half = nlon // 2
        self.lon_idx = longitude_idx

    def make_scalar(self, at, halo):
        nlon_half = self.nlon_half
        lon_idx = self.lon_idx

        @numba.njit()
        def fill_halos(psi, n, sign):
            lon = psi[0][lon_idx]
            return at(*psi, sign, nlon_half * (-1 if lon > nlon_half else 1))  # TODO: sign will not work for halo>2
        return fill_halos

    def make_vector(self, at):
        @numba.njit()
        def fill_halos(_, __, ___):
            raise NotImplementedError()
        return fill_halos

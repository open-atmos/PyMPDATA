"""
Created at 20.03.2020
"""

import numba


class PolarBoundaryCondition:
    def __init__(self, grid, longitude_idx, latitude_idx):
        self.nlon = grid[longitude_idx]
        self.nlat = grid[latitude_idx]
        assert self.nlon % 2 == 1

        self.nlon_half = self.nlon // 2 + 1
        self.lon_idx = longitude_idx
        self.lat_idx = latitude_idx

    def make_scalar(self, at, halo):
        nlon_half = self.nlon_half
        nlat = self.nlat
        lon_idx = self.lon_idx
        lat_idx = self.lat_idx
        self.halo = halo
        left_edge_idx = halo - 1
        right_edge_idx = nlat + halo

        @numba.njit()
        def fill_halos(psi, n, sign):
            lon = psi[0][lon_idx]
            lat = psi[0][lat_idx]
            if lat <= left_edge_idx:
                step = (left_edge_idx - lat) * 2 + 1
            if lat >= right_edge_idx:
                step = (lat - right_edge_idx) * 2 + 1

            # print(psi[0], step)

            return at(*psi, sign * step,
                      nlon_half * (-1 if lon > nlon_half else 1))  # TODO: sign will not work for halo>2

        return fill_halos

    def make_vector(self, at):
        @numba.njit()
        def fill_halos(_, __, ___):
            return 0

        return fill_halos

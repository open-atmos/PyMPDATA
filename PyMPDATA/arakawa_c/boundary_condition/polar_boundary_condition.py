import numba
from ..enumerations import ARG_FOCUS, SIGN_LEFT, SIGN_RIGHT


class PolarBoundaryCondition:
    def __init__(self, grid, longitude_idx, latitude_idx):
        assert SIGN_RIGHT == -1
        assert SIGN_LEFT == +1

        self.nlon = grid[longitude_idx]
        self.nlat = grid[latitude_idx]
        assert self.nlon % 2 == 0

        self.nlon_half = self.nlon // 2
        self.lon_idx = longitude_idx
        self.lat_idx = latitude_idx

    def make_scalar(self, at, halo):
        nlon_half = self.nlon_half
        nlat = self.nlat
        lon_idx = self.lon_idx
        lat_idx = self.lat_idx
        left_edge_idx = halo - 1
        right_edge_idx = nlat + halo

        @numba.njit()
        def fill_halos(psi, _, sign):
            lon = psi[ARG_FOCUS][lon_idx]
            lat = psi[ARG_FOCUS][lat_idx]
            if lat <= left_edge_idx:
                step = (left_edge_idx - lat) * 2 + 1
            if lat >= right_edge_idx:
                step = (lat - right_edge_idx) * 2 + 1

            val = nlon_half * (-1 if lon > nlon_half else 1)
            return at(*psi, sign * step, val)

        return fill_halos

    def make_vector(self, at):
        @numba.njit()
        def fill_halos(psi, _, __):
            return at(*psi, 0)  # TODO #120

        return fill_halos

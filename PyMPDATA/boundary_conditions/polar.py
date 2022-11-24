""" polar boundary condition for use in with spherical coordinates """
from functools import lru_cache

import numba

from PyMPDATA.impl.enumerations import ARG_FOCUS, SIGN_LEFT, SIGN_RIGHT


class Polar:
    """class which instances are to be passed in boundary_conditions tuple to the
    `ScalarField` and `VectorField` __init__ methods"""

    def __init__(self, grid, longitude_idx, latitude_idx):
        assert SIGN_RIGHT == -1
        assert SIGN_LEFT == +1

        self.nlon = grid[longitude_idx]
        self.nlat = grid[latitude_idx]
        assert self.nlon % 2 == 0

        self.nlon_half = self.nlon // 2
        self.lon_idx = longitude_idx
        self.lat_idx = latitude_idx

    def make_scalar(self, ats, halo, _, jit_flags):
        """returns (lru-cached) Numba-compiled scalar halo-filling callable"""
        nlon_half = self.nlon_half
        nlat = self.nlat
        lon_idx = self.lon_idx
        lat_idx = self.lat_idx
        left_edge_idx = halo - 1
        right_edge_idx = nlat + halo

        @numba.njit(**jit_flags)
        def fill_halos(psi, _, sign):
            lon = psi[ARG_FOCUS][lon_idx]
            lat = psi[ARG_FOCUS][lat_idx]
            if lat <= left_edge_idx:
                step = (left_edge_idx - lat) * 2 + 1
            else:
                step = (lat - right_edge_idx) * 2 + 1

            val = nlon_half * (-1 if lon > nlon_half else 1)
            return ats(*psi, sign * step, val)

        return fill_halos

    @staticmethod
    def make_vector(ats, _, __, jit_flags):
        """returns (lru-cached) Numba-compiled vector halo-filling callable"""
        return _make_vector_polar(ats, jit_flags)


@lru_cache()
def _make_vector_polar(ats, jit_flags):
    @numba.njit(**jit_flags)
    def fill_halos(psi, ___, ____):
        return ats(*psi, 0)  # TODO #120

    return fill_halos

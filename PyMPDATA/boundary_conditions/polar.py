""" polar boundary condition for use in with spherical coordinates """
from functools import lru_cache

import numba

from PyMPDATA.impl.enumerations import ARG_FOCUS, SIGN_LEFT, SIGN_RIGHT
from PyMPDATA.impl.traversals_common import (
    make_fill_halos_loop,
    make_fill_halos_loop_vector,
)


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

    def make_scalar(self, indexers, halo, _, jit_flags, dimension_index):
        """returns (lru-cached) Numba-compiled scalar halo-filling callable"""
        nlon_half = self.nlon_half
        nlat = self.nlat
        lon_idx = self.lon_idx
        lat_idx = self.lat_idx
        left_edge_idx = halo - 1
        right_edge_idx = nlat + halo
        ats = indexers.ats[dimension_index]
        set_value = indexers.set

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

        return make_fill_halos_loop(jit_flags, set_value, fill_halos)

    @staticmethod
    def make_vector(indexers, _, __, jit_flags, dimension_index):
        """returns (lru-cached) Numba-compiled vector halo-filling callable"""
        return _make_vector_polar(
            indexers.atv, indexers.set, jit_flags, dimension_index
        )


@lru_cache()
def _make_vector_polar(_atv, set_value, jit_flags, dimension_index):
    @numba.njit(**jit_flags)
    def fill_halos_parallel(_1, _2, _3):
        return 0  # TODO #120

    @numba.njit(**jit_flags)
    def fill_halos_normal(_1, _2, _3, _4):
        return 0  # TODO #120

    return make_fill_halos_loop_vector(
        jit_flags, set_value, fill_halos_parallel, fill_halos_normal, dimension_index
    )

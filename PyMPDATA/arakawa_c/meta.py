import numpy as np
from .enumerations import INNER, MID3D, OUTER

META_HALO_VALID = 0
META_N_OUTER = 1
META_N_MID3D = 2
META_N_INNER = 3
META_IS_NULL = 4
META_SIZE = 5


def make_meta(halo_valid: bool, grid):
    meta = np.empty(META_SIZE, dtype=int)
    meta[META_HALO_VALID] = halo_valid
    meta[META_N_OUTER] = grid[OUTER] if len(grid) > 1 else 0
    meta[META_N_MID3D] = grid[MID3D] if len(grid) > 2 else 0
    meta[META_N_INNER] = grid[INNER]
    meta[META_IS_NULL] = False
    return meta

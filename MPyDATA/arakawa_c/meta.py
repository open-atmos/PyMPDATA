import numpy as np

META_HALO_VALID = 0
META_N_OUTER = 1
META_N_INNER = 2
META_SIZE = 3


def make_meta(halo_valid: bool, grid):
    meta = np.empty(META_SIZE, dtype=int)
    meta[META_HALO_VALID] = halo_valid
    meta[META_N_OUTER] = grid[0] if len(grid) > 1 else 0
    meta[META_N_INNER] = grid[-1]
    return meta

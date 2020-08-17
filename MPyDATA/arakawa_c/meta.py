import numpy as np

meta_halo_valid = 0
meta_nouter = 1
meta_ninner = 2
meta_size = 3


def make_meta(halo_valid: bool, grid):
    meta = np.empty(meta_size, dtype=int)
    meta[meta_halo_valid] = halo_valid
    meta[meta_nouter] = grid[0] if len(grid) > 1 else 0
    meta[meta_ninner] = grid[-1]
    return meta
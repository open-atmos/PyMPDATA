import numpy as np

meta_halo_valid = 0
meta_ni = 1
meta_nj = 2
meta_nk = 3
meta_size = 4


def make_meta(halo_valid: bool, grid):
    meta = np.empty(meta_size, dtype=int)
    meta[meta_halo_valid] = halo_valid
    meta[meta_ni] = grid[0]
    meta[meta_nj] = grid[1] if len(grid) > 1 else 0
    meta[meta_nj] = grid[2] if len(grid) > 2 else 0
    return meta

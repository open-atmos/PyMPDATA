import numpy as np


class arakawa_c:
    @staticmethod
    def z_scalar_coord(grid):
        zZ = np.linspace(1 / 2, grid[0] - 1 / 2, grid[0])
        return zZ

    @staticmethod
    def z_vector_coord(grid):
        zZ = np.linspace(0, grid[0], grid[0] + 1)
        return zZ

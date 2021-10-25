""" common logic for `ScalarField` and `VectorField` classes """
from PyMPDATA.boundary_conditions.constant import Constant
from .meta import make_meta
from .enumerations import MAX_DIM_NUM, OUTER, MID3D, INNER, INVALID_HALO_VALUE


class Field:
    """ base class """
    def __init__(self, *, grid, boundary_conditions, halo, dtype):
        self.grid = grid
        self.meta = make_meta(False, grid)
        self.n_dims = len(grid)
        self.halo = halo
        self.dtype = dtype

        assert len(grid) == len(boundary_conditions)

        self.fill_halos = [None] * MAX_DIM_NUM
        self.fill_halos[OUTER] = boundary_conditions[OUTER] \
            if self.n_dims > 1 else Constant(INVALID_HALO_VALUE)
        self.fill_halos[MID3D] = boundary_conditions[MID3D] \
            if self.n_dims > 2 else Constant(INVALID_HALO_VALUE)
        self.fill_halos[INNER] = boundary_conditions[INNER]

        self.boundary_conditions = boundary_conditions

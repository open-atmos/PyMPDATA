"""
Created at 03.2020
"""

import numpy as np
from .indexers import make_null, indexers, MAX_DIM_NUM
from .scalar_field import ScalarField
from .traversals import make_meta, meta_halo_valid
from ..arakawa_c.boundary_condition.constant import Constant


class VectorField:
    def __init__(self, data, halo, boundary_conditions):
        self.halo = halo
        self.n_dims = len(data)

        dims = range(self.n_dims)
        halos = [[(halo - (d == c)) for c in dims] for d in dims]
        shape_with_halo = [[data[d].shape[c] + 2 * halos[d][c] for c in dims] for d in dims]
        self.data = [np.full(shape_with_halo[d], np.nan, dtype=np.float64) for d in dims]
        self.domain = tuple([tuple([slice(halos[d][c], halos[d][c] + data[d].shape[c]) for c in dims]) for d in dims])
        for d in dims:
            self.get_component(d)[:] = data[d][:]
        self.boundary_conditions = boundary_conditions
        self.fill_halos = tuple(
            [(boundary_conditions[i] if i < self.n_dims else Constant(np.nan)).make_vector(indexers[self.n_dims].at[i])
             for i in range(MAX_DIM_NUM)])
        grid = tuple([data[d].shape[d] - 1 for d in dims])
        self.meta = make_meta(False, grid)
        self.comp_0 = self.data[0]
        self.comp_1 = self.data[1] if self.n_dims > 1 else make_null()

    @staticmethod
    def clone(field):
        return VectorField([field.get_component(d) for d in range(field.n_dims)], field.halo, field.boundary_conditions)

    def get_component(self, i: int) -> np.ndarray:
        return self.data[i][self.domain[i]]

    def div(self, grid_step: tuple) -> ScalarField:
        diff_sum = None
        for d in range(self.n_dims):
            tmp = np.diff(self.get_component(d), axis=d) / grid_step[d]
            if diff_sum is None:
                diff_sum = tmp
            else:
                diff_sum += tmp
        result = ScalarField(diff_sum, halo=0, boundary_conditions=[Constant(np.nan)]*len(grid_step))
        return result

    @property
    def impl(self):
        return (self.meta, self.comp_0, self.comp_1), self.fill_halos

    @staticmethod
    def make_null(n_dims):
        null = VectorField([np.empty([0] * n_dims)] * n_dims, halo=1, boundary_conditions=[Constant(np.nan)]*n_dims)
        null.meta[meta_halo_valid] = True
        return null

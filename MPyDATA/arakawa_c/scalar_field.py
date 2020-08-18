"""
Created at 03.2020
"""

import numpy as np
from .indexers import indexers
from .enumerations import MAX_DIM_NUM
from .meta import META_HALO_VALID, make_meta
from ..arakawa_c.boundary_condition.constant_boundary_condition import ConstantBoundaryCondition
import inspect


class ScalarField:
    def __init__(self, data: np.ndarray, halo: int, boundary_conditions):
        assert len(data.shape) == len(boundary_conditions)
        for dim_length in data.shape:
            assert halo <= dim_length
        for bc in boundary_conditions:
            assert not inspect.isclass(bc)

        self.n_dims = data.ndim
        shape_with_halo = [data.shape[i] + 2 * halo for i in range(self.n_dims)]
        self.data = np.zeros(shape_with_halo, dtype=data.dtype)
        self.dtype = data.dtype
        self.halo = halo
        self.domain = tuple([slice(self.halo, self.data.shape[i] - self.halo) for i in range(self.n_dims)])
        self.get()[:] = data[:]

        fill_halos = [ConstantBoundaryCondition(np.nan)] * (MAX_DIM_NUM - self.n_dims)
        for d in range(self.n_dims):
            fill_halos.append(boundary_conditions[d])
        self.fill_halos = tuple([fh.make_scalar(indexers[self.n_dims].at[i], halo) for i, fh in enumerate(fill_halos)])

        self.boundary_conditions = boundary_conditions
        self.meta = make_meta(False, data.shape)

    @staticmethod
    def clone(field):
        return ScalarField(field.get(), field.halo, field.boundary_conditions)

    def get(self) -> np.ndarray:
        results = self.data[self.domain]
        return results

    @property
    def impl(self):
        return (self.meta, self.data), self.fill_halos

    @staticmethod
    def make_null(n_dims):
        null = ScalarField(np.empty([0]*n_dims), halo=0, boundary_conditions=[ConstantBoundaryCondition(np.nan)] * n_dims)
        null.meta[META_HALO_VALID] = True
        return null

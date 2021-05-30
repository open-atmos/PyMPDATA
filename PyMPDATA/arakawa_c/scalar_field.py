import numpy as np
from .indexers import indexers
from .enumerations import MAX_DIM_NUM, OUTER, MID3D, INNER, INVALID_HALO_VALUE, INVALID_INIT_VALUE, INVALID_NULL_VALUE
from .meta import META_HALO_VALID, make_meta, META_IS_NULL
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
        self.data = np.full(shape_with_halo, INVALID_INIT_VALUE, dtype=data.dtype)
        self.dtype = data.dtype
        self.halo = halo
        self.domain = tuple([slice(self.halo, self.data.shape[i] - self.halo) for i in range(self.n_dims)])
        self.get()[:] = data[:]

        # TODO #226
        fill_halos = [None] * MAX_DIM_NUM
        fill_halos[OUTER] = boundary_conditions[OUTER] if self.n_dims > 1 else ConstantBoundaryCondition(INVALID_HALO_VALUE)
        fill_halos[MID3D] = boundary_conditions[MID3D] if self.n_dims > 2 else ConstantBoundaryCondition(INVALID_HALO_VALUE)
        fill_halos[INNER] = boundary_conditions[INNER]
        self.fill_halos = tuple([fh.make_scalar(indexers[self.n_dims].at[i], halo) for i, fh in enumerate(fill_halos)])

        self.boundary_conditions = boundary_conditions
        self.meta = make_meta(False, data.shape)

    @staticmethod
    def clone(field):
        return ScalarField(field.get(), field.halo, field.boundary_conditions)

    def get(self) -> np.ndarray:  # note: actually a view is returned
        results = self.data[self.domain]
        return results

    @property
    def impl(self):
        return (self.meta, self.data), self.fill_halos

    @staticmethod
    def make_null(n_dims):
        null = ScalarField(np.empty([INVALID_NULL_VALUE]*n_dims), halo=0, boundary_conditions=[ConstantBoundaryCondition(np.nan)] * n_dims)
        null.meta[META_HALO_VALID] = True
        null.meta[META_IS_NULL] = True
        return null

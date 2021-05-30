import numpy as np
from .indexers import indexers
from .enumerations import MAX_DIM_NUM, OUTER, MID3D, INNER, INVALID_NULL_VALUE, INVALID_INIT_VALUE, INVALID_HALO_VALUE
from .scalar_field import ScalarField
from .meta import META_HALO_VALID, make_meta, META_IS_NULL
from ..arakawa_c.boundary_condition.constant_boundary_condition import ConstantBoundaryCondition
import inspect


class VectorField:
    def __init__(self, data, halo, boundary_conditions):
        assert len(data) == len(boundary_conditions)
        for field in data:
            assert len(field.shape) == len(data)
            for dim_length in field.shape:
                assert halo <= dim_length
        for bc in boundary_conditions:
            assert not inspect.isclass(bc)

        self.halo = halo
        self.n_dims = len(data)
        self.dtype = data[0].dtype

        dims = range(self.n_dims)
        halos = [[(halo - (d == c)) for c in dims] for d in dims]
        shape_with_halo = [[data[d].shape[c] + 2 * halos[d][c] for c in dims] for d in dims]
        self.data = [np.full(shape_with_halo[d], INVALID_INIT_VALUE, dtype=self.dtype) for d in dims]
        self.domain = tuple([tuple([slice(halos[d][c], halos[d][c] + data[d].shape[c]) for c in dims]) for d in dims])
        for d in dims:
            assert data[d].dtype == self.dtype
            self.get_component(d)[:] = data[d][:]
        self.boundary_conditions = boundary_conditions

        # TODO #226: code repeated in ScalarField
        fill_halos = [None] * MAX_DIM_NUM
        fill_halos[OUTER] = boundary_conditions[OUTER] if self.n_dims > 1 else ConstantBoundaryCondition(INVALID_HALO_VALUE)
        fill_halos[MID3D] = boundary_conditions[MID3D] if self.n_dims > 2 else ConstantBoundaryCondition(INVALID_HALO_VALUE)
        fill_halos[INNER] = boundary_conditions[INNER]
        self.fill_halos = tuple([fh.make_vector(indexers[self.n_dims].at[i]) for i, fh in enumerate(fill_halos)])

        grid = tuple([data[d].shape[d] - 1 for d in dims])
        self.meta = make_meta(False, grid)
        self.comp_outer = self.data[0] if self.n_dims > 1 else np.empty(tuple([0] * self.n_dims), dtype=self.dtype)
        self.comp_mid3d = self.data[1] if self.n_dims > 2 else np.empty(tuple([0] * self.n_dims), dtype=self.dtype)
        self.comp_inner = self.data[-1]

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
        result = ScalarField(diff_sum, halo=0, boundary_conditions=[ConstantBoundaryCondition(INVALID_HALO_VALUE)] * len(grid_step))
        return result

    @property
    def impl(self):
        return (self.meta, self.comp_outer, self.comp_mid3d, self.comp_inner), self.fill_halos

    @staticmethod
    def make_null(n_dims):
        null = VectorField(
            [np.full([1] * n_dims, INVALID_NULL_VALUE)] * n_dims,
            halo=1,
            boundary_conditions=[ConstantBoundaryCondition(INVALID_HALO_VALUE)] * n_dims
        )
        null.meta[META_HALO_VALID] = True
        null.meta[META_IS_NULL] = True
        return null

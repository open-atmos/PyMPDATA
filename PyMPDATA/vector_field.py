"""
vector field abstractions for the staggered grid
"""
import inspect
import numpy as np
from PyMPDATA.impl.enumerations import INVALID_NULL_VALUE, INVALID_INIT_VALUE, INVALID_HALO_VALUE, \
    INNER, OUTER, MID3D
from PyMPDATA.scalar_field import ScalarField
from PyMPDATA.impl.meta import META_HALO_VALID, META_IS_NULL
from PyMPDATA.boundary_conditions.constant import Constant
from PyMPDATA.impl.field import Field


class VectorField(Field):
    """ n-component n-dimensional vector field including halo data """

    def __init__(self, data: tuple, halo: int, boundary_conditions: tuple):
        super().__init__(
            grid=tuple(data[d].shape[d] - 1 for d, _ in enumerate(data)),
            boundary_conditions=boundary_conditions,
            halo=halo,
            dtype=data[0].dtype
        )

        for comp, field in enumerate(data):
            assert len(field.shape) == len(data)
            for dim, dim_length in enumerate(field.shape):
                assert halo <= dim_length
                if not (np.asarray(self.grid) == 0).all():
                    assert dim_length == self.grid[dim] + (dim == comp)
        for boundary_condition in boundary_conditions:
            assert not inspect.isclass(boundary_condition)

        dims = range(self.n_dims)
        halos = tuple(tuple((halo - (dim == comp)) for comp in dims) for dim in dims)
        shape_with_halo = tuple(
            tuple(data[dim].shape[comp] + 2 * halos[dim][comp] for comp in dims)
            for dim in dims
        )
        self.data = tuple(
            np.full(shape_with_halo[dim], INVALID_INIT_VALUE, dtype=self.dtype)
            for dim in dims
        )
        self.domain = tuple(
            tuple(
                slice(halos[dim][comp], halos[dim][comp] + data[dim].shape[comp])
                for comp in dims)
            for dim in dims
        )
        for dim in dims:
            assert data[dim].dtype == self.dtype
            self.get_component(dim)[:] = data[dim][:]

        empty = np.empty(tuple([0] * self.n_dims), dtype=self.dtype)
        self.impl_data = (
            self.data[OUTER] if self.n_dims > 1 else empty,
            self.data[MID3D] if self.n_dims > 2 else empty,
            self.data[INNER]
        )

    @staticmethod
    def clone(field):
        return VectorField(tuple(
            field.get_component(d)
            for d in range(field.n_dims)
        ), field.halo, field.boundary_conditions)

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
        result = ScalarField(
            diff_sum,
            halo=0,
            boundary_conditions=tuple([Constant(INVALID_HALO_VALUE)] * len(grid_step))
        )
        return result

    @staticmethod
    def make_null(n_dims, indexers):
        null = VectorField(
            [np.full([1] * n_dims, INVALID_NULL_VALUE)] * n_dims,
            halo=1,
            boundary_conditions=tuple([Constant(INVALID_HALO_VALUE)] * n_dims)
        )
        null.meta[META_HALO_VALID] = True
        null.meta[META_IS_NULL] = True
        null.assemble(indexers)
        return null

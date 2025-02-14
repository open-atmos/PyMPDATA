"""
vector field abstractions for the staggered grid
"""

import inspect

import numpy as np

from PyMPDATA.boundary_conditions.constant import Constant
from PyMPDATA.impl.enumerations import (
    INNER,
    INVALID_HALO_VALUE,
    INVALID_INIT_VALUE,
    INVALID_NULL_VALUE,
    MID3D,
    OUTER,
    IMPL_META_AND_DATA,
    IMPL_BC,
)
from PyMPDATA.impl.field import Field
from PyMPDATA.scalar_field import ScalarField


class VectorField(Field):
    """n-component n-dimensional vector field including halo data,
    used to represent the advector field"""

    def __init__(self, data: tuple, halo: int, boundary_conditions: tuple):
        super().__init__(
            grid=tuple(datum.shape[dim] - 1 for dim, datum in enumerate(data)),
            boundary_conditions=boundary_conditions,
            halo=halo,
            dtype=data[0].dtype,
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
                for comp in dims
            )
            for dim in dims
        )
        for dim in dims:
            assert data[dim].dtype == self.dtype
            self.get_component(dim)[:] = data[dim][:]

        empty = np.empty(tuple([0] * self.n_dims), dtype=self.dtype)
        self._impl_data = (
            self.data[OUTER] if self.n_dims > 1 else empty,
            self.data[MID3D] if self.n_dims > 2 else empty,
            self.data[INNER],
        )

    @staticmethod
    def clone(field):
        """returns a newly allocated field of the same shape, halo and boundary conditions"""
        return VectorField(
            tuple(field.get_component(d) for d in range(field.n_dims)),
            field.halo,
            field.boundary_conditions,
        )

    def get_component(self, i: int) -> np.ndarray:
        """returns a view over given component of the field excluding halo"""
        return self.data[i][self.domain[i]]

    def div(self, grid_step: tuple) -> ScalarField:
        """returns a newly allocated scalar field (with no halo) containing the
        divergence of the vector field computed using minimal stencil"""
        diff_sum = None
        for dim in range(self.n_dims):
            tmp = np.diff(self.get_component(dim), axis=dim) / grid_step[dim]
            if diff_sum is None:
                diff_sum = tmp
            else:
                diff_sum += tmp
        result = ScalarField(
            diff_sum,
            halo=0,
            boundary_conditions=tuple([Constant(INVALID_HALO_VALUE)] * len(grid_step)),
        )
        return result

    @staticmethod
    def make_null(n_dims, traversals):
        """returns a vector field instance with no allocated data storage,
        see `Field._make_null` other properties of the returned field"""
        return Field._make_null(
            VectorField(
                tuple([np.full([1] * n_dims, INVALID_NULL_VALUE)] * n_dims),
                halo=1,
                boundary_conditions=tuple([Constant(INVALID_HALO_VALUE)] * n_dims),
            ),
            traversals,
        )
    def _fill_halos(self, traversals):
        f = traversals._code["fill_halos_vector"]
        #TODO: assert n_threads == 1
        buf = traversals.data.buffer
        f(0, self.impl[IMPL_META_AND_DATA][0], (
            self.impl[IMPL_META_AND_DATA][1],
            self.impl[IMPL_META_AND_DATA][2],
            self.impl[IMPL_META_AND_DATA][3],
        ), self.impl[IMPL_BC], buf)

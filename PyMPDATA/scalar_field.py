"""
scalar field abstractions for the staggered grid
"""

import inspect

import numpy as np

from PyMPDATA.boundary_conditions import Constant
from PyMPDATA.impl.enumerations import INVALID_INIT_VALUE, IMPL_META_AND_DATA, IMPL_BC
from PyMPDATA.impl.field import Field


class ScalarField(Field):
    """n-dimensional scalar field including halo data, used to represent advectee, g_factor, etc."""

    def __init__(self, data: np.ndarray, halo: int, boundary_conditions: tuple):
        super().__init__(
            grid=data.shape,
            boundary_conditions=boundary_conditions,
            halo=halo,
            dtype=data.dtype,
        )

        for dim_length in data.shape:
            assert halo <= dim_length
        for boundary_condition in boundary_conditions:
            assert not inspect.isclass(boundary_condition)

        shape_with_halo = tuple(data.shape[i] + 2 * halo for i in range(self.n_dims))
        self.data = np.full(shape_with_halo, INVALID_INIT_VALUE, dtype=data.dtype)
        self._impl_data = (self.data,)
        self.domain = tuple(
            slice(halo, self.data.shape[i] - halo) for i in range(self.n_dims)
        )
        self.get()[:] = data[:]

    @staticmethod
    def clone(field, dtype=None):
        """returns an instance with the same dimensionality and same halo size as the argument
        optionally with a different data type"""
        dtype = dtype or field.dtype
        # note: copy=False is OK as the ctor anyhow copies the data to an array with halo
        return ScalarField(
            field.get().astype(dtype, copy=False), field.halo, field.boundary_conditions
        )

    def get(self) -> np.ndarray:  # note: actually a view is returned
        """returns a view (not a copy) of the field data excluding halo"""
        results = self.data[self.domain]
        return results

    @staticmethod
    def make_null(n_dims, traversals):
        """returns a scalar field instance with no allocated data storage,
        see `Field._make_null` other properties of the returned field"""
        return Field._make_null(
            ScalarField(
                np.empty(shape=[0] * n_dims),
                halo=0,
                boundary_conditions=tuple([Constant(np.nan)] * n_dims),
            ),
            traversals,
        )
    def _fill_halos(self, traversals):
        f = traversals._code["fill_halos_scalar"]
        buf = traversals.data.buffer
        #TODO: assert n_threads == 1
        f(0, *self.impl[IMPL_META_AND_DATA], self.impl[IMPL_BC], buf)

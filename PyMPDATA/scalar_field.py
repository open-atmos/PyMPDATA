import numpy as np
from PyMPDATA.impl.enumerations import INVALID_INIT_VALUE, INVALID_NULL_VALUE
from PyMPDATA.impl.meta import META_HALO_VALID, META_IS_NULL
from PyMPDATA.boundary_conditions import Constant
from PyMPDATA.impl.field import Field
import inspect


class ScalarField(Field):
    def __init__(self, data: np.ndarray, halo: int, boundary_conditions):
        super().__init__(grid=data.shape, boundary_conditions=boundary_conditions)

        for dim_length in data.shape:
            assert halo <= dim_length
        for bc in boundary_conditions:
            assert not inspect.isclass(bc)

        shape_with_halo = [data.shape[i] + 2 * halo for i in range(self.n_dims)]
        self.data = np.full(shape_with_halo, INVALID_INIT_VALUE, dtype=data.dtype)
        self.dtype = data.dtype
        self.halo = halo
        self.domain = tuple(
            slice(self.halo, self.data.shape[i] - self.halo)
            for i in range(self.n_dims)
        )
        self.get()[:] = data[:]

        self.impl = None
        self.jit_flags = None

    def assemble(self, traversals):
        if traversals.jit_flags != self.jit_flags:
            self.impl = (self.meta, self.data), tuple(
                fh.make_scalar(
                    traversals.indexers[self.n_dims].at[i],
                    self.halo,
                    self.dtype,
                    traversals.jit_flags
                )
                for i, fh in enumerate(self.fill_halos)
            )
        self.jit_flags = traversals.jit_flags

    @staticmethod
    def clone(field, dtype=None):
        dtype = dtype or field.dtype
        # note: copy=False is OK as the ctor anyhow copies the data to an array with halo
        return ScalarField(
            field.get().astype(dtype, copy=False),
            field.halo,
            field.boundary_conditions
        )

    def get(self) -> np.ndarray:  # note: actually a view is returned
        results = self.data[self.domain]
        return results

    @staticmethod
    def make_null(n_dims, traversals):
        null = ScalarField(
            np.empty([INVALID_NULL_VALUE]*n_dims),
            halo=0,
            boundary_conditions=[Constant(np.nan)] * n_dims
        )
        null.meta[META_HALO_VALID] = True
        null.meta[META_IS_NULL] = True
        null.assemble(traversals)
        return null

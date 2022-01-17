""" common logic for `PyMPDATA.scalar_field.ScalarField` and
 `PyMPDATA.vector_field.VectorField` classes """
from collections import namedtuple
import abc
from PyMPDATA.boundary_conditions.constant import Constant
from .meta import make_meta
from .enumerations import MAX_DIM_NUM, OUTER, MID3D, INNER, INVALID_HALO_VALUE
from .meta import META_IS_NULL, META_HALO_VALID


_Properties = namedtuple(
    '__Properties',
    ('grid', 'meta', 'n_dims', 'halo', 'dtype', 'boundary_conditions')
)


class Field:
    """ abstract base class for scalar and vector fields """
    def __init__(self, *, grid: tuple, boundary_conditions: tuple, halo: int, dtype):
        assert len(grid) <= len(boundary_conditions)

        self.fill_halos = [None] * MAX_DIM_NUM
        self.fill_halos[OUTER] = boundary_conditions[OUTER] \
            if len(boundary_conditions) > 1 else Constant(INVALID_HALO_VALUE)
        self.fill_halos[MID3D] = boundary_conditions[MID3D] \
            if len(boundary_conditions) > 2 else Constant(INVALID_HALO_VALUE)
        self.fill_halos[INNER] = boundary_conditions[INNER]

        self.__properties = _Properties(
            grid=grid,
            meta=make_meta(False, grid),
            n_dims=len(grid),
            halo=halo,
            dtype=dtype,
            boundary_conditions=self.fill_halos
        )

        self.__impl = None
        self.__jit_flags = None
        self._impl_data = None

    @property
    def n_dims(self):
        """ dimensionality """
        return self.__properties.n_dims

    @property
    def halo(self):
        """ halo extent (in each dimension), for vector fields the staggered dimension
            of each component has the extent equal to halo-1 """
        return self.__properties.halo

    @property
    def dtype(self):
        """ data type (e.g., np.float64) """
        return self.__properties.dtype

    @property
    def grid(self):
        """ tuple defining grid geometry without halo (same interpretation as np.ndarray.shape) """
        return self.__properties.grid

    @property
    def meta(self):
        """ tuple encoding meta data abount the scalar field (e.g., if halo was filled, ...) """
        return self.__properties.meta

    @property
    def impl(self):
        """ tuple combining meta, data and boundary conditions - for passing to njit-ted code """
        return self.__impl

    @property
    def boundary_conditions(self):
        """ tuple of boundary conditions as passed to the `__init__()` (plus Constant(NaN) in
            dimensions higher than grid diemensionality) """
        return self.__properties.boundary_conditions

    @property
    def jit_flags(self):
        """ jit_flags used in the last call to assemble() """
        return self.__jit_flags

    def assemble(self, traversals):
        """ initialises what can be later accessed through `PyMPDATA.impl.field.Field.impl` property
            with halo-filling logic njit-ted using the given traversals """
        if traversals.jit_flags != self.__jit_flags:
            fun = f'make_{self.__class__.__name__[:6].lower()}'
            self.__impl = (self.__properties.meta, *self._impl_data), tuple(
                getattr(fh, fun)(
                    traversals.indexers[self.n_dims].ats[i],
                    self.halo,
                    self.dtype,
                    traversals.jit_flags
                )
                for i, fh in enumerate(self.fill_halos)
            )
        self.__jit_flags = traversals.jit_flags

    @staticmethod
    def _make_null(null_field, traversals):
        null_field.meta[META_HALO_VALID] = True
        null_field.meta[META_IS_NULL] = True
        null_field.assemble(traversals)
        return null_field

    @staticmethod
    @abc.abstractmethod
    def make_null(n_dims: int, traversals):  # pylint: disable=missing-function-docstring
        raise NotImplementedError()

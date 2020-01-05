from .impl.field import Field
from .scalar_field import ScalarField
from .impl.vector_field_2d import make_vector_field_2d
from .impl.vector_field_1d import make_vector_field_1d
import numpy as np


class VectorField(Field):
    def __init__(self, data, halo, boundary_conditions, impl=None):
        if impl is None:
            dimension = len(data)
            if dimension == 1:
                self._impl = make_vector_field_1d(data[0], halo) if impl is None else impl
            elif dimension == 2:
                self._impl = make_vector_field_2d(data, halo) if impl is None else impl
            elif dimension == 3:
                raise NotImplementedError()
            else:
                raise ValueError()
        else:
            self._impl = impl

        self.boundary_conditions = boundary_conditions
        super().__init__(halo)

    def add(self, rhs):
        for d in range(self.dimension):
            self.get_component(d)[:] += rhs.get_component(d)[:]
        self._halo_valid = False

    def div(self, grid_step: tuple) -> ScalarField:
        diffsum = None
        for d in range(self.dimension):
            tmp = np.diff(self.get_component(d), axis=d) / grid_step[d]
            if diffsum is None:
                diffsum = tmp
            else:
                diffsum += tmp
        result = ScalarField(diffsum, halo=0, boundary_conditions=None)
        return result

    def get_component(self, i: int) -> np.ndarray:
        return self._impl.get_component(i)

    def clone(self):
        return VectorField(
            data=None,
            halo=self.halo,
            boundary_conditions=self.boundary_conditions,
            impl=self._impl.clone()
        )

    def _fill_halos_impl(self):
        if self.dimension == 1 and self.halo == 1:
            return
        for axis in range(self.dimension):
            for comp in range(self.dimension):
                if self.dimension == 2 and self.halo < 2 and comp == axis:
                    continue
                for side in (0, 1):
                    self.boundary_conditions[comp][side].vector(self._impl, axis, comp)

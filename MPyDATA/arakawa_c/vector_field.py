from .impl.field import Field
from .scalar_field import ScalarField
from .impl.vector_field_2d import make_vector_field_2d
from .impl.vector_field_1d import make_vector_field_1d
import numpy as np


class VectorField(Field):
    def __init__(self, data: iter, halo: int, boundary_conditions, impl=None):
        if len(data) == 1:
            super().__init__(halo, (data[0].shape[0]-1,))
            self._impl = make_vector_field_1d(data[0], halo) if impl is None else impl
        elif len(data) == 2:
            super().__init__(halo, (data[1].shape[0], data[0].shape[1]))
            self._impl = make_vector_field_2d(data, halo) if impl is None else impl
        elif len(data) == 3:
            raise NotImplementedError()
        else:
            raise ValueError()

        self.boundary_conditions = boundary_conditions

    def div(self, grid_step: tuple) -> ScalarField:
        result = ScalarField(np.zeros(self.grid), halo=0, boundary_conditions=None)
        for d in range(self.dimension):
            result.get()[:] += np.diff(self.get_component(d), axis=d) / grid_step[d]
        return result

    def get_component(self, i: int) -> np.ndarray:
        return self._impl.get_component(i)

    @staticmethod
    def full_like(vector_field, value: float = np.nan):
        data = [np.full_like(vector_field.get_component(d), value) for d in range(vector_field.dimension)]
        return VectorField(
            data=data,
            halo=vector_field.halo,
            boundary_conditions=vector_field.boundary_conditions,
            impl=vector_field._impl.clone()
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

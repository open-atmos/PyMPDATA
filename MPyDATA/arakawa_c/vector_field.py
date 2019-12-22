from .impl.field import Field
from .scalar_field import ScalarField
from .impl.vector_field_2d import VectorField2D
from .impl.vector_field_1d import VectorField1D
import numpy as np


class VectorField(Field):
    def __init__(self, data: iter, halo: int):
        if len(data) == 1:
            self._impl = VectorField1D(data[0], halo)
        elif len(data) == 2:
            self._impl = VectorField2D(data[0], data[1], halo)
        elif len(data) == 3:
            raise NotImplementedError()
        else:
            raise ValueError()

    def div(self, grid_step: tuple) -> ScalarField:
        result = ScalarField(np.zeros(self.shape), halo=0)
        for d in range(self.dimension):
            result.get()[:] += np.diff(self.get_component(d), axis=d) / grid_step[d]
        return result

    def get_component(self, i: int) -> np.ndarray:
        return self._impl.get_component(i)

    @staticmethod
    def full_like(vector_field, value: float = np.nan):
        data = [np.full_like(vector_field.get_component(d), value) for d in range(vector_field.dimension)]
        return VectorField(data, halo=vector_field.halo)


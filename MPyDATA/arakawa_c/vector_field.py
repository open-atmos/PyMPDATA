from MPyDATA.arakawa_c._field import __Field
from MPyDATA.arakawa_c._impl._vector_field_2d import _VectorField2D
from MPyDATA.arakawa_c._impl._vector_field_1d import _VectorField1D
import numpy as np


class VectorField(__Field):
    def __init__(self, data: iter, halo: int):
        if len(data) == 1:
            self._impl = _VectorField1D(data[0], halo)
        elif len(data) == 2:
            self._impl = _VectorField2D(data[0], data[1], halo)
        elif len(data) == 3:
            raise NotImplementedError()
        else:
            raise ValueError()

    def get_component(self, i: int) -> np.ndarray:
        return self._impl.get_component(i)

    @staticmethod
    def full_like(vector_field, value: float = np.nan):
        data = [np.full_like(vector_field.get_component(d), value) for d in range(vector_field.dimension)]
        return VectorField(data, halo=vector_field.halo)


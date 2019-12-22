from .impl.field import Field
from .impl.scalar_field_1d import ScalarField1D
from .impl.scalar_field_2d import ScalarField2D
import numpy as np


class ScalarField(Field):
    def __init__(self, data, halo):
        dimension = len(data.shape)

        if dimension == 1:
            self._impl = ScalarField1D(data, halo)
        elif dimension == 2:
            self._impl = ScalarField2D(data, halo)
        elif dimension == 3:
            raise NotImplementedError()
        else:
            raise ValueError()

    def add(self, rhs):
        self._impl.get()[:] += rhs._impl.get()[:]
        self._impl.invalidate_halos()

    def get(self) -> np.ndarray:
        return self._impl.get()

    @staticmethod
    def full_like(scalar_field, value: float = np.nan):
        return ScalarField(data=np.full_like(scalar_field.get(), value), halo=scalar_field.halo)


from MPyDATA.arakawa_c._field import __Field
from MPyDATA.arakawa_c._impl._scalar_field_1d import _ScalarField1D
from MPyDATA.arakawa_c._impl._scalar_field_2d import _ScalarField2D
import numpy as np


class ScalarField(__Field):
    class Impl:
        def at(self, i: int, j: int=-1, k: int=-1):
            raise NotImplementedError()

    def __init__(self, data, halo):
        dimension = len(data.shape)

        if dimension == 1:
            self._impl = _ScalarField1D(data, halo)
        elif dimension == 2:
            self._impl = _ScalarField2D(data, halo)
        elif dimension == 3:
            raise NotImplementedError()
        else:
            raise ValueError()

    def get(self) -> np.ndarray:
        return self._impl.get()

    @staticmethod
    def full_like(scalar_field, value: float = np.nan):
        return ScalarField(data=np.full_like(scalar_field.get(), value), halo=scalar_field.halo)


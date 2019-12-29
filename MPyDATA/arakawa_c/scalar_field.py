from .impl.field import Field
from .impl.scalar_field_1d import make_scalar_field_1d
from .impl.scalar_field_2d import make_scalar_field_2d
import numpy as np


class ScalarField(Field):
    def __init__(self, data, halo, boundary_conditions):
        super().__init__(halo, data.shape)
        dimension = len(data.shape)

        if dimension == 1:
            self._impl = make_scalar_field_1d(data, halo)
        elif dimension == 2:
            self._impl = make_scalar_field_2d(data, halo)
        elif dimension == 3:
            raise NotImplementedError()
        else:
            raise ValueError()

        self.boundary_conditions = boundary_conditions

    def add(self, rhs):
        self.get()[:] += rhs.get()[:]
        self._halo_valid = False

    def get(self) -> np.ndarray:
        return self._impl.get()

    @staticmethod
    def full_like(scalar_field, value: float = np.nan):
        return ScalarField(
            data=np.full_like(scalar_field.get(), value),
            halo=scalar_field.halo,
            boundary_conditions=scalar_field.boundary_conditions
        )

    def _fill_halos_impl(self):
        for d in range(self.dimension):
            for side in (0, 1):
                self.boundary_conditions[d][side].scalar(self._impl, d)

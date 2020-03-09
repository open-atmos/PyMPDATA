from .impl.field import Field
from .impl.scalar_field_2d import make_scalar_field_2d
import numpy as np


class ScalarField(Field):
    def __init__(self, data, halo, boundary_conditions, impl=None):
        if impl is None:
            dimension = len(data.shape)
            if dimension == 2:
                self._impl = make_scalar_field_2d(data, halo)
            elif dimension == 3:
                raise NotImplementedError()
            else:
                raise ValueError()
        else:
            self._impl = impl

        self.boundary_conditions = boundary_conditions
        super().__init__(halo)

    def add(self, rhs):
        self.get()[:] += rhs.get()[:]
        self._halo_valid = False

    def get(self) -> np.ndarray:
        return self._impl.get()

    def clone(self):
        return ScalarField(
            data=None,
            halo=self.halo,
            boundary_conditions=self.boundary_conditions,
            impl=self._impl.clone()
        )

    def _fill_halos_impl(self):
        for d in range(self.dimension):
            for side in (0, 1):
                self.boundary_conditions[d][side].scalar(self._impl, d)

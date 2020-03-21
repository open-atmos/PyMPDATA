import numpy as np
from .utils import make_null, make_flag, indexers
from .scalar_field import ScalarField
from ..arakawa_c.boundary_condition.cyclic import Cyclic


class VectorField:
    def __init__(self, data, halo, boundary_conditions=(Cyclic, Cyclic)):
        self.halo = halo
        self.n_dims = len(data)

        dims = range(self.n_dims)
        halos = [[(halo - (d == c)) for c in dims] for d in dims]
        shape_with_halo = [[data[d].shape[c] + 2 * halos[d][c] for c in dims] for d in dims]
        self.data = [np.full(shape_with_halo[d], np.nan, dtype=np.float64) for d in dims]
        self.domain = tuple([tuple([slice(halos[d][c], halos[d][c] + data[d].shape[c]) for c in dims]) for d in dims])
        for d in dims:
            self.get_component(d)[:] = data[d][:]
        self.boundary_conditions = (  # TODO: list comprehension?
            boundary_conditions[0].make_vector(indexers[self.n_dims].at0),
            boundary_conditions[1].make_vector(indexers[self.n_dims].at1),
        )
        self.flag = make_flag(False)

    @staticmethod
    def clone(field):
        return VectorField([field.get_component(d) for d in range(field.n_dims)], field.halo)

    def get_component(self, i: int) -> np.ndarray:
        return self.data[i][self.domain[i]]

    def div(self, grid_step: tuple) -> ScalarField:
        diff_sum = None
        for d in range(self.n_dims):
            tmp = np.diff(self.get_component(d), axis=d) / grid_step[d]
            if diff_sum is None:
                diff_sum = tmp
            else:
                diff_sum += tmp
        result = ScalarField(diff_sum, halo=0)
        return result

    @property
    def impl(self):
        comp_0 = self.data[0]
        comp_1 = self.data[1] if self.n_dims > 1 else make_null()
        return (self.flag, comp_0, comp_1), self.boundary_conditions

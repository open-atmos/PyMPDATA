import numpy as np
from .scalar_field import ScalarField


class VectorField:
    def __init__(self, data, halo):
        self.halo = halo
        self.n_dims = len(data)

        dims = range(self.n_dims)
        halos = [[(halo - (d == c)) for c in dims] for d in dims]
        shape_with_halo = [[data[d].shape[c] + 2 * halos[d][c] for c in dims] for d in dims]
        self.data = [np.full(shape_with_halo[d], np.nan, dtype=np.float64) for d in dims]
        self.domain = [[slice(halos[d][c], halos[d][c] + data[d].shape[c]) for c in dims] for d in dims]
        for d in dims:
            self.get_component(d)[:] = data[d][:]

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

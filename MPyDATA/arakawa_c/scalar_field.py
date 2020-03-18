import numpy as np
from .utils import make_flag


class ScalarField:
    def __init__(self, data, halo):
        self.n_dims = len(data.shape)
        shape_with_halo = [data.shape[i] + 2 * halo for i in range(self.n_dims)]
        self.data = np.zeros(shape_with_halo, dtype=np.float64)
        self.halo = halo
        self.domain = tuple([slice(self.halo, self.data.shape[i] - self.halo) for i in range(self.n_dims)])
        self.get()[:] = data[:]

    def get(self) -> np.ndarray:
        results = self.data[self.domain]
        return results

    @property
    def impl(self):
        return make_flag(False), self.data

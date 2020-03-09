import numpy as np

from ..formulae.halo import halo


class ScalarField:
    def __init__(self, data):
        self.shape = (data.shape[0] + 2 * halo, data.shape[1] + 2 * halo)

        self.data = np.zeros((self.shape[0], self.shape[1]), dtype=np.float64)
        self.get()[:, :] = data[:, :]

    def get(self) -> np.ndarray:
        results = self.data[
                  halo: self.data.shape[0] - halo,
                  halo: self.data.shape[1] - halo
                  ]
        return results

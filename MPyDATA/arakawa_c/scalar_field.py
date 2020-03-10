import numpy as np


class ScalarField:
    def __init__(self, data, halo):
        self.shape = (data.shape[0] + 2 * halo, data.shape[1] + 2 * halo)
        self.halo = halo
        self.data = np.zeros((self.shape[0], self.shape[1]), dtype=np.float64)
        self.get()[:, :] = data[:, :]

    def get(self) -> np.ndarray:
        results = self.data[
                  self.halo: self.data.shape[0] - self.halo,
                  self.halo: self.data.shape[1] - self.halo
                  ]
        return results

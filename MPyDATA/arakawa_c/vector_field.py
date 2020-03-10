import numpy as np
from .scalar_field import ScalarField


class VectorField:
    def __init__(self, data_0, data_1, halo):
        self.halo = halo
        self.shape = (data_1.shape[0], data_0.shape[1])
        self.dimension = len(self.shape)

        self.data_0 = np.full((
            data_0.shape[0] + 2 * (halo - 1),
            data_0.shape[1] + 2 * halo
        ), np.nan, dtype=np.float64)
        self.data_1 = np.full((
            data_1.shape[0] + 2 * halo,
            data_1.shape[1] + 2 * (halo - 1)
        ), np.nan, dtype=np.float64)

        self.get_component(0)[:, :] = data_0[:, :]
        self.get_component(1)[:, :] = data_1[:, :]

    @staticmethod
    def clone(field):
        return VectorField(field.get_component(0), field.get_component(1), field.halo)

    def get_component(self, i: int) -> np.ndarray:
        domain = (
            slice(
                self.halo - 1,
                self.halo - 1 + self.shape[0] + 1
            ),
            slice(
                self.halo,
                self.halo + self.shape[1]
            )
        ) if i == 0 else (
            slice(
                self.halo,
                self.halo + self.shape[0]
            ),
            slice(
                self.halo - 1,
                self.halo - 1 + self.shape[1] + 1
            )
        )
        if i == 0:
            return self.data_0[domain]
        elif i == 1:
            return self.data_1[domain]

    def div(self, grid_step: tuple) -> ScalarField:
        diff_sum = None
        for d in range(self.dimension):
            tmp = np.diff(self.get_component(d), axis=d) / grid_step[d]
            if diff_sum is None:
                diff_sum = tmp
            else:
                diff_sum += tmp
        result = ScalarField(diff_sum, halo=0)
        return result

import numpy as np
from ..formulae.halo import halo


class VectorField:
    def __init__(self, data_0, data_1):
        self.shape = (data_1.shape[0], data_0.shape[1])

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
        return VectorField(field.get_component(0), field.get_component(1))

    def get_component(self, i: int) -> np.ndarray:
        domain = (
            slice(
                halo - 1,
                halo - 1 + self.shape[0] + 1
            ),
            slice(
                halo,
                halo + self.shape[1]
            )
        ) if i == 0 else (
            slice(
                halo,
                halo + self.shape[0]
            ),
            slice(
                halo - 1,
                halo - 1 + self.shape[1] + 1
            )
        )
        if i == 0:
            return self.data_0[domain]
        elif i == 1:
            return self.data_1[domain]
# pylint: disable=invalid-name, missing-module-docstring, missing-class-docstring, missing-function-docstring, too-few-public-methods
import numpy as np


class Settings:
    courant_field = -0.5, -0.25
    output_steps = (0,)  # TODO: 75)

    @staticmethod
    def initial_condition(xi, yi, grid):
        nx, ny = grid
        return np.exp(
            -((xi + 0.5 - nx / 2) ** 2) / (2 * (nx / 10) ** 2)
            - (yi + 0.5 - ny / 2) ** 2 / (2 * (ny / 10) ** 2)
        )

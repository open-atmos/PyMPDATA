from MPyDATA.arakawa_c.scalar_field import ScalarField
from MPyDATA.arakawa_c.vector_field import VectorField
import numpy as np


def div(vector_field: VectorField, grid_step: tuple) -> ScalarField:
    if vector_field.dimension == 2:
        return _div_2d(vector_field, grid_step)
    raise NotImplementedError()


def _div_2d(vector_field: VectorField, grid_step: tuple) -> ScalarField:
    result = ScalarField(np.zeros(vector_field.shape), halo=0)
    for d in range(vector_field.dimension):
        result.get()[:] += np.diff(vector_field.get_component(d), axis=d) / grid_step[d]
    return result

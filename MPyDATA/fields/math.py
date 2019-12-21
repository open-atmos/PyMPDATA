from MPyDATA.fields.interfaces import IScalarField, IVectorField
from MPyDATA.fields.factories import make_scalar_field
import numpy as np


def div(vector_field: IVectorField, grid_step: tuple) -> IScalarField:
    if vector_field.dimension == 2:
        return _div_2d(vector_field, grid_step)
    raise NotImplementedError()


def _div_2d(vector_field: IVectorField, grid_step: tuple) -> IScalarField:
    result = make_scalar_field(np.zeros(vector_field.shape), halo=0)
    for d in range(vector_field.dimension):
        result.data[:, :] += np.diff(vector_field.get_component(d), axis=d) / grid_step[d]
    return result

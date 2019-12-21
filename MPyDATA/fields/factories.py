from MPyDATA.fields._scalar_field_1d import _ScalarField1D
from MPyDATA.fields._scalar_field_2d import _ScalarField2D
from MPyDATA.fields._vector_field_1d import _VectorField1D
from MPyDATA.fields._vector_field_2d import _VectorField2D
from MPyDATA.fields.interfaces import IScalarField, IVectorField

import numpy as np


def make_scalar_field(data, halo):
    dimension = len(data.shape)

    if dimension == 1:
        return _ScalarField1D(data, halo)
    if dimension == 2:
        return _ScalarField2D(data, halo)
    if dimension == 3:
        raise NotImplementedError()
    raise ValueError()


def make_vector_field(data: iter, halo: int):
    if len(data) == 1:
        return _VectorField1D(data[0], halo)
    if len(data) == 2:
        return _VectorField2D(data[0], data[1], halo)
    if len(data) == 3:
        raise NotImplementedError()
    else:
        raise ValueError()


def clone_scalar_field(scalar_field: IScalarField, value: float = np.nan):
    return make_scalar_field(data=np.full_like(scalar_field.get(), value), halo=scalar_field.halo)


def clone_vector_field(vector_field: IVectorField, value: float = np.nan):
    data = [np.full_like(vector_field.get_component(d), value) for d in range(vector_field.dimension)]
    return make_vector_field(data, halo=vector_field.halo)


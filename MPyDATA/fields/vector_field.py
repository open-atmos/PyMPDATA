"""
Created at 02.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np
from MPyDATA.fields._vector_field_1d import VectorField1D
from MPyDATA.fields._vector_field_2d import VectorField2D, div_2d


class Interface:
    def at(self, i, j): raise NotImplementedError()
    def apply(self, function, arg_1, arg_2): raise NotImplementedError()
    def fill_halos(self): raise NotImplementedError()
    # TODO: ...


def make(data, halo):
    if len(data) == 1:
        return VectorField1D(data[0], halo)
    if len(data) == 2:
        return VectorField2D(data[0], data[1], halo)
    if len(data) == 3:
        raise NotImplementedError()
    else:
        raise ValueError()


def clone(vector_field, value=np.nan):
    data = [np.full_like(vector_field.get_component(d), value) for d in range(vector_field.dimension)]
    return make(data, halo=vector_field.halo)


def apply(function, output, args: tuple, ext=0):
    assert len(args) == 2
    for arg in args:
        arg.fill_halos()
    output.apply_2arg(function, args[0], args[1], ext)
    output.invalidate_halos()


def div(vector_field, grid_step: tuple):
    if vector_field.dimension == 2:
        return div_2d(vector_field, grid_step)
    raise NotImplementedError()


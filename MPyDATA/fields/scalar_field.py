"""
Created at 02.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from MPyDATA.fields._scalar_field_1d import ScalarField1D
from MPyDATA.fields._scalar_field_2d import ScalarField2D
import numpy as np


class Interface:
    def at(self, i, j): raise NotImplementedError()
    def swap_memory(self, other): raise NotImplementedError()
    def fill_halos(self): raise NotImplementedError()
    def get(self): raise NotImplementedError()
    data: np.array
    halo: int
    # TODO: ...


# TODO: rename to make
def make(data, halo):
    dimension = len(data.shape)

    if dimension == 1:
        return ScalarField1D(data, halo)
    if dimension == 2:
        return ScalarField2D(data, halo)
    if dimension == 3:
        raise NotImplementedError()
    raise ValueError()


def clone(scalar_field):
    data = scalar_field.get()
    return make(data=data, halo=scalar_field.halo)


def apply(function, output, args, ext=0):
    assert ext < output.halo
    if len(args) == 1:
        output.apply_1arg(function, args[0], ext)
    elif len(args) == 2:
        output.apply_2arg(function, args[0], args[1], ext)
    elif len(args) == 4:
        output.apply_4arg(function, args[0], args[1], args[2], args[3], ext)
    else:
        raise NotImplementedError()
"""
Created at 11.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from MPyDATA.fields.scalar_field import ScalarField
from MPyDATA.fields.vector_field import VectorField
from MPyDATA.opts import Opts
import numpy as np
import numba

# TODO
HALO = 1


def make_flux(opts: Opts, it: int):
    iga = opts.iga

    @numba.njit()
    def flux(psi: ScalarField, GC: VectorField):
        if it == 0 or not iga:
            result = (
                np.maximum(0, GC.at(+.5, 0)) * psi.at(0, 0) +
                np.minimum(0, GC.at(+.5, 0)) * psi.at(1, 0)
            )
        else:
            result = GC.at(+.5, 0)
        return result
    # TODO: check if (abs(c)-C)/2 is not faster
    return flux




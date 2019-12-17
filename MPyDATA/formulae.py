"""
Created at 11.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from MPyDATA.fields.scalar_field import ScalarField
from MPyDATA.fields.vector_field import VectorField
import numpy as np
import numba


EPS = 1e-8
HALO = 1


def make_flux(iga, it):
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


@numba.njit()
def upwind(flx: VectorField, G: ScalarField):
    return - 1/G.at(0, 0) * (
            flx.at(+.5, 0) -
            flx.at(-.5, 0)
    )


# TODO: G!
def make_antidiff(iga):
    @numba.njit()
    def antidiff(psi: ScalarField, C: VectorField):
        # TODO comment
        def A(psi):
            result = psi.at(1, 0) - psi.at(0, 0)
            if iga:
                result /= 2
            else:
                result /= (psi.at(1, 0) + psi.at(0, 0) + EPS)
            return result

        result = (np.abs(C.at(+.5, 0)) - C.at(+.5, 0) ** 2) * A(psi)

        for i in range(len(psi.shape)):
            if i == psi.axis:
                continue
            result -= 0.5 * C.at(+.5, 0) * 0.25 * (C.at(1, +.5) + C.at(0, +.5) + C.at(1, -.5) + C.at(0, -.5)) * \
                      (psi.at(1, 1) + psi.at(0, 1) - psi.at(1, -1) - psi.at(0, -1))
            if iga:
                result /= 4
            else:
                result /= (psi.at(1, 1) + psi.at(0, 1) + psi.at(1, -1) + psi.at(0, -1) + EPS)
            # TODO dx, dt
        return result
    return antidiff




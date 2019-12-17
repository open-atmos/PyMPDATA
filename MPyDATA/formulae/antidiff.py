"""
Created at 17.12.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from MPyDATA.fields.scalar_field import ScalarField
from MPyDATA.fields.vector_field import VectorField
from MPyDATA.opts import Opts
import numpy as np
import numba


# TODO: G!
def make_antidiff(opts: Opts):
    iga = opts.iga
    eps = opts.eps

    @numba.njit()
    def antidiff(psi: ScalarField, C: VectorField):
        # TODO comment
        def A(psi):
            result = psi.at(1, 0) - psi.at(0, 0)
            if iga:
                result /= 2
            else:
                result /= (psi.at(1, 0) + psi.at(0, 0) + eps)
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
                result /= (psi.at(1, 1) + psi.at(0, 1) + psi.at(1, -1) + psi.at(0, -1) + eps)
            # TODO dx, dt
        return result
    return antidiff




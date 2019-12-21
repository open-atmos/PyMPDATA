"""
Created at 17.12.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from MPyDATA.options import Options
from MPyDATA.fields.interfaces import IScalarField, IVectorField
import numpy as np

from MPyDATA_tests.utils import debug
if debug.DEBUG:
    import MPyDATA_tests.utils.fake_numba as numba
else:
    import numba


# TODO: G!
def make_antidiff(opts: Options):
    iga = opts.iga
    eps = opts.eps

    @numba.njit()
    def antidiff(psi: IScalarField, C: IVectorField):
        # eq. 13 in Smolarkiewicz 1984; eq. 17a in Smolarkiewicz & Margolin 1998
        def A(psi):
            result = psi.at(1, 0) - psi.at(0, 0)
            if iga:
                result /= 2
            else:
                result /= (psi.at(1, 0) + psi.at(0, 0) + eps)
            return result

        # eq. 13 in Smolarkiewicz 1984; eq. 17b in Smolarkiewicz & Margolin 1998
        def B(psi):
            result = (
                psi.at(1, 1) + psi.at(0, 1) -
                psi.at(1, -1) - psi.at(0, -1)
            )
            if iga:
                result /= 4
            else:
                result /= (
                    psi.at(1, 1) + psi.at(0, 1) +
                    psi.at(1, -1) + psi.at(0, -1) +
                    eps
                )
            return result

        # eq. 13 in Smolarkiewicz 1984
        result = (np.abs(C.at(+.5, 0)) - C.at(+.5, 0) ** 2) * A(psi)
        for i in range(len(psi.shape)):
            if i == psi.axis:
                continue
            result -= (
                0.5 * C.at(+.5, 0) *
                0.25 * (C.at(1, +.5) + C.at(0, +.5) + C.at(1, -.5) + C.at(0, -.5)) *
                B(psi)
            )
        return result
    return antidiff




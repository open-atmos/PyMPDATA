"""
Created at 17.12.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from ..arakawa_c.scalar_field import ScalarField
from ..arakawa_c.vector_field import VectorField
import numpy as np

from ..utils import debug_flag

if debug_flag.VALUE:
    import MPyDATA.utils.fake_numba as numba
else:
    import numba


# TODO: G!
def make_antidiff(opts):
    iga = opts.iga
    eps = opts.eps
    dfl = opts.dfl
    tot = opts.tot
    nug = opts.nug

    @numba.njit
    def antidiff(psi: ScalarField.Impl, GC: VectorField.Impl, G: ScalarField.Impl):
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
        result = (np.abs(GC.at(+.5, 0)) - GC.at(+.5, 0) ** 2) * A(psi)
        for i in range(psi.dimension):
            if i == psi.axis:
                continue
            result -= (
                    0.5 * GC.at(+.5, 0) *
                    0.25 * (GC.at(1, +.5) + GC.at(0, +.5) + GC.at(1, -.5) + GC.at(0, -.5)) *
                    B(psi)
            )

        # third-order terms
        if tot:
            assert psi.dimension == 1  # TODO!
            # TODO: if nug
            tmp = 2 * (psi.at(2,0) - psi.at(1,0) - psi.at(0,0) + psi.at(-1,0)) * (
                     3 * GC.at(.5,0) * np.abs(GC.at(.5,0)) / ((G.at(1,0) + G.at(0,0)) / 2)
                     - 2 * GC.at(.5,0) ** 3 / ((G.at(1,0) + G.at(0,0)) / 2) ** 2
                     - GC.at(.5,0)
             ) / 6

            if iga:
                tmp /= (1 + 1 + 1 + 1)
            else:
                tmp /= (psi.at(2,0) + psi.at(1,0) + psi.at(0,0) + psi.at(-1,0))

            result += tmp

        # divergent flow option
        if dfl:
            assert psi.dimension == 1  # TODO!
            tmp = -.5 * GC.at(.5, 0) * (GC.at(1.5, 0) - GC.at(-.5, 0))
            if nug:
                tmp /= (G.at(1, 0) + G.at(0, 0))
            if iga:
                tmp *= .5 * (psi.at(1, 0) + psi.at(0, 0))
            result += tmp
        return result
    return antidiff




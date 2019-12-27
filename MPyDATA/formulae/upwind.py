"""
Created at 11.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from ..options import Options
from ..arakawa_c.scalar_field import ScalarField
from ..arakawa_c.vector_field import VectorField

from ..utils import debug_flag

if debug_flag.VALUE:
    import MPyDATA.utils.fake_numba as numba
else:
    import numba


def make_upwind(opts: Options):
    nug = opts.nug

    @numba.njit
    def upwind(flx: VectorField.Impl, G: ScalarField.Impl):
        result = - 1 * (
                flx.at(+.5, 0) -
                flx.at(-.5, 0)
        )
        if not nug:
            result /= G.at(0, 0)
        return result
    return upwind

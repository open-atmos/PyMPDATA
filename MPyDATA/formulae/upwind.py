"""
Created at 11.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from MPyDATA.fields.interfaces import IScalarField, IVectorField
from MPyDATA.options import Options
from MPyDATA_tests.utils import debug

if debug.DEBUG:
    import MPyDATA_tests.utils.fake_numba as numba
else:
    import numba


def make_upwind(opts: Options):
    nug = opts.nug

    @numba.njit()
    def upwind(flx: IVectorField, G:IScalarField):
        result = - 1 * (
                flx.at(+.5, 0) -
                flx.at(-.5, 0)
        )
        if not nug:
            result /= G.at(0, 0)
        return result
    return upwind

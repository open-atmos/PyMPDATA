"""
Created at 11.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from MPyDATA.fields import scalar_field
from MPyDATA.fields import vector_field
from MPyDATA.opts import Opts
import numba


def make_upwind(opts: Opts):
    nug = opts.nug

    @numba.njit()
    def upwind(flx: vector_field.Interface, G:scalar_field.Interface):
        result = - 1 * (
                flx.at(+.5, 0) -
                flx.at(-.5, 0)
        )
        if not nug:
            result /= G.at(0, 0)
        return result
    return upwind

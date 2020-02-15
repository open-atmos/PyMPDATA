"""
Created at 11.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from ..arakawa_c.scalar_field import ScalarField
from ..arakawa_c.vector_field import VectorField
from ..arakawa_c.traversal import Traversal
from ..utils import debug_flag
from .jit_flags import jit_flags

if debug_flag.VALUE:
    import MPyDATA.utils.fake_numba as numba
else:
    import numba


def make_upwind(opts):
    nug = opts.nug

    @numba.njit(**jit_flags)
    def upwind(init: float, flx: VectorField.Impl, G: ScalarField.Impl):
        result = -1 * (
                flx.at(+.5, 0) -
                flx.at(-.5, 0)
        )
        if nug:
            result /= G.at(0, 0)
        return init + result
    return Traversal(body=upwind, init=0, loop=True)

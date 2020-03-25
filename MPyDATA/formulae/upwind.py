"""
Created at 11.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from MPyDATA.jit_flags import jit_flags
import numba
from MPyDATA.arakawa_c.utils import indexers, MAX_DIM_NUM


def make_upwind(non_unit_g_factor, traversals):
    apply_scalar = traversals.apply_scalar(loop=True)
    idx = indexers[traversals.n_dims]

    formulae_upwind = tuple([__make_upwind(idx.atv[i], idx.at[i], non_unit_g_factor) for i in range(MAX_DIM_NUM)])

    @numba.njit(**jit_flags)
    def apply(psi, flux, vec_bc, g_factor, g_factor_bc):
        return apply_scalar(*formulae_upwind, *psi, *flux, *vec_bc, *g_factor, *g_factor_bc)

    return apply


def __make_upwind(atv, at, nug):
    @numba.njit(**jit_flags)
    def upwind(init, flux, g_factor):
        result = \
               + atv(*flux, -.5, 0) \
               - atv(*flux, .5, 0)
        if nug:
            result /= at(*g_factor, 0, 0)
        return init + result
    return upwind

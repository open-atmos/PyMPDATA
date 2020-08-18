"""
Created at 11.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numba
from MPyDATA.arakawa_c.enumerations import MAX_DIM_NUM
from MPyDATA.arakawa_c.indexers import indexers


def make_upwind(options, non_unit_g_factor, traversals):
    apply_scalar = traversals.apply_scalar(loop=True)
    idx = indexers[traversals.n_dims]

    formulae_upwind = tuple([
        __make_upwind(options.jit_flags, idx.atv[i], idx.at[i], non_unit_g_factor)
        if i >= MAX_DIM_NUM - traversals.n_dims else None
        for i in range(MAX_DIM_NUM)
    ])

    @numba.njit(**options.jit_flags)
    def apply(psi, flux, vec_bc, g_factor, g_factor_bc):
        null_scalfield = g_factor
        null_scalfield_bc = g_factor_bc
        return apply_scalar(*formulae_upwind, *psi, *flux, *vec_bc, *g_factor, *g_factor_bc,
                            *null_scalfield, *null_scalfield_bc, *null_scalfield, *null_scalfield_bc)

    return apply


def __make_upwind(jit_flags, atv, at, nug):
    @numba.njit(**jit_flags)
    def upwind(init, flux, g_factor, _, __):
        result = \
               + atv(*flux, -.5) \
               - atv(*flux, .5)
        if nug:
            result /= at(*g_factor, 0)
        return init + result
    return upwind

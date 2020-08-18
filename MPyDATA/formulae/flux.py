"""
Created at 11.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np
import numba
from MPyDATA.arakawa_c.indexers import indexers
from MPyDATA.arakawa_c.enumerations import MAX_DIM_NUM


def make_flux_first_pass(options, traversals):
    idx = indexers[traversals.n_dims]
    apply_vector = traversals.apply_vector()

    formulae_flux_first_pass = tuple([
        __make_flux(options.jit_flags, idx.atv[i], idx.at[i], first_pass=True, infinite_gauge=False)
        if i >= MAX_DIM_NUM - traversals.n_dims else None
        for i in range(MAX_DIM_NUM)
    ])

    @numba.njit(**options.jit_flags)
    def apply(vectmp_a, advector, advectee, advectee_bc, vec_bc):
        null_scalarfield = advectee
        null_bc = vec_bc
        return apply_vector(*formulae_flux_first_pass, *vectmp_a, *advectee, *advectee_bc, *advector, *vec_bc,
                            *null_scalarfield, *null_bc)

    return apply


def make_flux_subsequent(options, traversals):
    if options.n_iters <= 1:
        @numba.njit(**options.jit_flags)
        def apply(_flux, _psi, _psi_bc, _GC_corr, _vec_bc):
            return
    else:
        idx = indexers[traversals.n_dims]
        apply_vector = traversals.apply_vector()

        formulae_flux_subsequent = tuple([
            __make_flux(options.jit_flags, idx.atv[i], idx.at[i], first_pass=False, infinite_gauge=options.infinite_gauge)
            if i >= MAX_DIM_NUM - traversals.n_dims else None
            for i in range(MAX_DIM_NUM)
        ])

        @numba.njit(**options.jit_flags)
        def apply(flux, psi, psi_bc, GC_corr, vec_bc):
            null_scalarfield = psi
            null_bc = vec_bc
            return apply_vector(*formulae_flux_subsequent, *flux, *psi, *psi_bc, *GC_corr, *vec_bc,
                                *null_scalarfield, *null_bc)

    return apply


def __make_flux(jit_flags, atv, at, first_pass, infinite_gauge):
    @numba.njit(**jit_flags)
    def minimum_0(c):
        return (c - np.abs(c)) / 2

    @numba.njit(**jit_flags)
    def maximum_0(c):
        return (c + np.abs(c)) / 2

    if not first_pass and infinite_gauge:
        @numba.njit(**jit_flags)
        def flux(_, advector, __):
            return atv(*advector, +.5, 0)
    else:
        @numba.njit(**jit_flags)
        def flux(advectee, advector, __):
            return \
                maximum_0(atv(*advector, +.5, 0)) * at(*advectee, 0, 0) + \
                minimum_0(atv(*advector, +.5, 0)) * at(*advectee, 1, 0)
    return flux

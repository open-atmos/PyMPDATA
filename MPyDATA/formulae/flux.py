"""
Created at 11.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np
import numba
from MPyDATA.arakawa_c.indexers import indexers, MAX_DIM_NUM
from MPyDATA.arakawa_c.traversals import null_vector_formula


def make_flux_first_pass(options, traversals):
    idx = indexers[traversals.n_dims]
    apply_vector = traversals.apply_vector()

    formulae_flux_first_pass = tuple([
        __make_flux(options.jit_flags, idx.atv[i], idx.at[i], first_pass=True, infinite_gauge=False)
        if i < traversals.n_dims else null_vector_formula
        for i in range(MAX_DIM_NUM)
    ])

    @numba.njit(**options.jit_flags)
    def apply(vectmp_a, GC_phys, psi, psi_bc, vec_bc):
        null_scalarfield = psi
        null_bc = vec_bc
        return apply_vector(*formulae_flux_first_pass, *vectmp_a, *psi, *psi_bc, *GC_phys, *vec_bc,
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

        formulae_flux_subsequent = (
            __make_flux(options.jit_flags, idx.atv[0], idx.at[0], first_pass=False, infinite_gauge=options.infinite_gauge),
            __make_flux(options.jit_flags, idx.atv[1], idx.at[1], first_pass=False, infinite_gauge=options.infinite_gauge)
        )

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
        def flux(_, GC, __):
            return atv(*GC, +.5, 0)
    else:
        @numba.njit(**jit_flags)
        def flux(psi, GC, __):
            return \
                maximum_0(atv(*GC, +.5, 0)) * at(*psi, 0, 0) + \
                minimum_0(atv(*GC, +.5, 0)) * at(*psi, 1, 0)
    return flux

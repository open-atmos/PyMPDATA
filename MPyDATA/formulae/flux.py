"""
Created at 11.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np
from MPyDATA.jit_flags import jit_flags
import numba
from MPyDATA.arakawa_c.utils import indexers, null_formula


@numba.njit(**jit_flags)
def minimum_0(c):
    return (c - np.abs(c)) / 2


@numba.njit(**jit_flags)
def maximum_0(c):
    return (c + np.abs(c)) / 2


def make_flux_first_pass(n_dims, apply_vector):
    idx = indexers[n_dims]

    formulae_flux_first_pass = (
        __make_flux_first_pass(idx.atv0, idx.at0),
        __make_flux_first_pass(idx.atv1, idx.at1),
        null_formula,
        null_formula
    )

    @numba.njit(**jit_flags)
    def apply(vectmp_a, GC_phys, psi, psi_bc, vec_bc):
        null_scalarfield = psi
        null_bc = vec_bc
        return apply_vector(False, *formulae_flux_first_pass, *vectmp_a, *psi, *psi_bc, *GC_phys, *vec_bc,
                            *null_scalarfield, *null_bc)

    return apply


def __make_flux_first_pass(atv, at):

    @numba.njit(**jit_flags)
    def flux(psi, GC, _):
        return \
            maximum_0(atv(*GC, +.5, 0)) * at(*psi, 0, 0) + \
            minimum_0(atv(*GC, +.5, 0)) * at(*psi, 1, 0)
    return flux


def make_flux_subsequent(n_dims, options, apply_vector):
    if options.n_iters <= 1:  # TODO: this s handled by the above make_flux_first_pass???
        @numba.njit(**jit_flags)
        def apply(_flux, _psi, _psi_bc, _GC_corr, _vec_bc):
            return
    else:
        idx = indexers[n_dims]

        formulae_flux_subsequent = (
            __make_flux_subsequent(idx.atv0, idx.at0, infinite_gauge=options.infinite_gauge),
            __make_flux_subsequent(idx.atv1, idx.at1, infinite_gauge=options.infinite_gauge),
            null_formula,
            null_formula
        )

        @numba.njit(**jit_flags)
        def apply(flux, psi, psi_bc, GC_corr, vec_bc):
            null_scalarfield = psi
            null_bc = vec_bc
            return apply_vector(False, *formulae_flux_subsequent, *flux, *psi, *psi_bc, *GC_corr, *vec_bc,
                                *null_scalarfield, *null_bc)

    return apply


def __make_flux_subsequent(atv, at, infinite_gauge):
    if infinite_gauge:
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

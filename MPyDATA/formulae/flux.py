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


def make_flux_first_pass(traversals):
    idx = indexers[traversals.n_dims]
    apply_vector = traversals.apply_vector(loop=False)

    formulae_flux_first_pass = (
        __make_flux(idx.atv0, idx.at0, first_pass=True, infinite_gauge=False),
        __make_flux(idx.atv1, idx.at1, first_pass=True, infinite_gauge=False),
        null_formula,
        null_formula
    )

    @numba.njit(**jit_flags)
    def apply(vectmp_a, GC_phys, psi, psi_bc, vec_bc):
        null_scalarfield = psi
        null_bc = vec_bc
        return apply_vector(*formulae_flux_first_pass, *vectmp_a, *psi, *psi_bc, *GC_phys, *vec_bc,
                            *null_scalarfield, *null_bc)

    return apply


def make_flux_subsequent(options, traversals):
    if options.n_iters <= 1:  # TODO: this s handled by the above make_flux_first_pass???
        @numba.njit(**jit_flags)
        def apply(_flux, _psi, _psi_bc, _GC_corr, _vec_bc):
            return
    else:
        idx = indexers[traversals.n_dims]
        apply_vector = traversals.apply_vector(loop=False)

        formulae_flux_subsequent = (
            __make_flux(idx.atv0, idx.at0, first_pass=False, infinite_gauge=options.infinite_gauge),
            __make_flux(idx.atv1, idx.at1, first_pass=False, infinite_gauge=options.infinite_gauge),
            null_formula,
            null_formula
        )

        @numba.njit(**jit_flags)
        def apply(flux, psi, psi_bc, GC_corr, vec_bc):
            null_scalarfield = psi
            null_bc = vec_bc
            return apply_vector(*formulae_flux_subsequent, *flux, *psi, *psi_bc, *GC_corr, *vec_bc,
                                *null_scalarfield, *null_bc)

    return apply


def __make_flux(atv, at, first_pass, infinite_gauge):
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

""" staggered-grid flux logic including infinite-gauge logic handling """
import numba
import numpy as np

from PyMPDATA.impl.enumerations import MAX_DIM_NUM


def make_flux_first_pass(options, traversals):
    """returns njit-ted function for use with given traversals"""
    idx = traversals.indexers[traversals.n_dims]
    apply_vector = traversals.apply_vector()

    formulae_flux_first_pass = tuple(
        __make_flux(
            options.jit_flags,
            idx.atv[i],
            idx.ats[i],
            first_pass=True,
            infinite_gauge=False,
        )
        if idx.ats[i] is not None
        else None
        for i in range(MAX_DIM_NUM)
    )

    @numba.njit(**options.jit_flags)
    def apply(null_impl, vectmp_a, advector, advectee):
        null_scalarfield, null_bc = null_impl.scalar
        return apply_vector(
            *formulae_flux_first_pass,
            *vectmp_a.field,
            *advectee.field,
            advectee.bc,
            *advector.field,
            advector.bc,
            *null_scalarfield,
            null_bc
        )

    return apply


def make_flux_subsequent(options, traversals):
    """returns njit-ted function for use with given traversals"""
    idx = traversals.indexers[traversals.n_dims]
    apply_vector = traversals.apply_vector()

    formulae_flux_subsequent = tuple(
        __make_flux(
            options.jit_flags,
            idx.atv[i],
            idx.ats[i],
            first_pass=False,
            infinite_gauge=options.infinite_gauge,
        )
        if idx.ats[i] is not None
        else None
        for i in range(MAX_DIM_NUM)
    )

    @numba.njit(**options.jit_flags)
    def apply(null_impl, flux, psi, g_c_corr):
        null_scalarfield, null_bc = null_impl.scalar
        return apply_vector(
            *formulae_flux_subsequent,
            *flux.field,
            *psi.field,
            psi.bc,
            *g_c_corr.field,
            g_c_corr.bc,
            *null_scalarfield,
            null_bc
        )

    return apply


def __make_flux(jit_flags, atv, ats, first_pass, infinite_gauge):
    @numba.njit(**jit_flags)
    def minimum_0(arg):
        return (arg - np.abs(arg)) / 2

    @numba.njit(**jit_flags)
    def maximum_0(arg):
        return (arg + np.abs(arg)) / 2

    if not first_pass and infinite_gauge:

        @numba.njit(**jit_flags)
        def flux(_, advector, __):
            return atv(*advector, +0.5)

    else:

        @numba.njit(**jit_flags)
        def flux(advectee, advector, __):
            return maximum_0(atv(*advector, +0.5)) * ats(*advectee, 0) + minimum_0(
                atv(*advector, +0.5)
            ) * ats(*advectee, 1)

    return flux

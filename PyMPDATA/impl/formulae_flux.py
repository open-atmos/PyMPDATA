import numpy as np
import numba
from PyMPDATA.impl.enumerations import MAX_DIM_NUM


def make_flux_first_pass(options, traversals):
    idx = traversals.indexers[traversals.n_dims]
    apply_vector = traversals.apply_vector()
    null_scalarfield, null_bc = traversals.null_scalar_field.impl

    formulae_flux_first_pass = tuple(
        __make_flux(
            options.jit_flags,
            idx.atv[i],
            idx.ats[i],
            first_pass=True,
            infinite_gauge=False
        )
        if idx.ats[i] is not None else None
        for i in range(MAX_DIM_NUM)
    )

    @numba.njit(**options.jit_flags)
    def apply(vectmp_a, advector, advectee, advectee_bc, vec_bc):
        return apply_vector(*formulae_flux_first_pass,
                            *vectmp_a,
                            *advectee, *advectee_bc,
                            *advector, *vec_bc,
                            *null_scalarfield, *null_bc)

    return apply


def make_flux_subsequent(options, traversals):
    idx = traversals.indexers[traversals.n_dims]
    apply_vector = traversals.apply_vector()

    formulae_flux_subsequent = tuple(
        __make_flux(
            options.jit_flags,
            idx.atv[i], idx.ats[i],
            first_pass=False,
            infinite_gauge=options.infinite_gauge
        )
        if idx.ats[i] is not None else None
        for i in range(MAX_DIM_NUM)
    )

    null_scalarfield, null_bc = traversals.null_scalar_field.impl

    @numba.njit(**options.jit_flags)
    def apply(flux, psi, psi_bc, GC_corr, vec_bc):
        return apply_vector(*formulae_flux_subsequent,
                            *flux,
                            *psi, *psi_bc,
                            *GC_corr, *vec_bc,
                            *null_scalarfield, *null_bc)

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
            return atv(*advector, +.5)
    else:
        @numba.njit(**jit_flags)
        def flux(advectee, advector, __):
            return \
                maximum_0(atv(*advector, +.5)) * ats(*advectee, 0) + \
                minimum_0(atv(*advector, +.5)) * ats(*advectee, 1)
    return flux

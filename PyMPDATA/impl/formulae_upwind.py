""" upwind/donor-cell formula logic including G-factor handling """
import numba

from PyMPDATA.impl.enumerations import MAX_DIM_NUM


def make_upwind(options, non_unit_g_factor, traversals):
    """returns an njit-ted function for use with given traversals"""
    apply_scalar = traversals.apply_scalar(loop=True)
    idx = traversals.indexers[traversals.n_dims]

    formulae_upwind = tuple(
        __make_upwind(options.jit_flags, idx.atv[i], idx.ats[i], non_unit_g_factor)
        if idx.ats[i] is not None
        else None
        for i in range(MAX_DIM_NUM)
    )

    @numba.njit(**options.jit_flags)
    def apply(traversals_data, psi, flux, g_factor):
        null_scalarfield, null_scalarfield_bc = traversals_data.null_scalar_field
        return apply_scalar(
            *formulae_upwind,
            *psi.field,
            *flux.field,
            flux.bc,
            *g_factor.field,
            g_factor.bc,
            *null_scalarfield,
            null_scalarfield_bc,
            *null_scalarfield,
            null_scalarfield_bc,
            *null_scalarfield,
            null_scalarfield_bc,
            traversals_data.buffer
        )

    return apply


def __make_upwind(jit_flags, atv, ats, nug):
    @numba.njit(**jit_flags)
    def upwind(init, flux, g_factor, _, __, ___):
        result = +atv(*flux, -0.5) - atv(*flux, 0.5)
        if nug:
            result /= ats(*g_factor, 0)
        return init + result

    return upwind

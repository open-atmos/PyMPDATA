import numba
from PyMPDATA.impl.enumerations import MAX_DIM_NUM


def make_upwind(options, non_unit_g_factor, traversals):
    apply_scalar = traversals.apply_scalar(loop=True)
    idx = traversals.indexers[traversals.n_dims]
    null_scalarfield, null_scalarfield_bc = traversals.null_scalar_field.impl

    formulae_upwind = tuple(
        __make_upwind(options.jit_flags, idx.atv[i], idx.ats[i], non_unit_g_factor)
        if idx.ats[i] is not None else None
        for i in range(MAX_DIM_NUM)
    )

    @numba.njit(**options.jit_flags)
    def apply(psi, flux, vec_bc, g_factor, g_factor_bc):
        return apply_scalar(*formulae_upwind,
                            *psi,
                            *flux, *vec_bc,
                            *g_factor, *g_factor_bc,
                            *null_scalarfield, *null_scalarfield_bc,
                            *null_scalarfield, *null_scalarfield_bc,
                            *null_scalarfield, *null_scalarfield_bc
                            )

    return apply


def __make_upwind(jit_flags, atv, ats, nug):
    @numba.njit(**jit_flags)
    def upwind(init, flux, g_factor, _, __, ___):
        result = \
               + atv(*flux, -.5) \
               - atv(*flux, .5)
        if nug:
            result /= ats(*g_factor, 0)
        return init + result
    return upwind

"""logic for handling the Fickian term by modifying physical velocity"""

import numba

from ..impl.enumerations import MAX_DIM_NUM
from ..impl.traversals import Traversals
from ..options import Options


def make_laplacian(non_unit_g_factor: bool, options: Options, traversals: Traversals):
    """returns njit-ted function for use with given traversals"""
    if not options.non_zero_mu_coeff:

        @numba.njit(**options.jit_flags)
        def apply(_1, _2, _3):
            return

    else:
        idx = traversals.indexers[traversals.n_dims]
        apply_vector = traversals.apply_vector()

        formulae_laplacian = tuple(
            (
                __make_laplacian(
                    options.jit_flags, idx.ats[i], options.epsilon, non_unit_g_factor
                )
                if idx.ats[i] is not None
                else None
            )
            for i in range(MAX_DIM_NUM)
        )

        @numba.njit(**options.jit_flags)
        def apply(traversals_data, advector, advectee):
            null_vecfield, null_vecfield_bc = traversals_data.null_vector_field
            null_scalarfield, null_scalarfield_bc = traversals_data.null_scalar_field
            return apply_vector(
                *formulae_laplacian,
                *advector.field,
                *advectee.field,
                advectee.bc,
                *null_vecfield,
                null_vecfield_bc,
                *null_scalarfield,
                null_scalarfield_bc,
                traversals_data.buffer
            )

    return apply


def __make_laplacian(jit_flags, ats, epsilon, non_unit_g_factor):
    if non_unit_g_factor:
        raise NotImplementedError()

    @numba.njit(**jit_flags)
    def fun(advectee, _, __):
        return (
            -2
            * (ats(*advectee, 1) - ats(*advectee, 0))
            / (ats(*advectee, 1) + ats(*advectee, 0) + epsilon)
        )

    return fun

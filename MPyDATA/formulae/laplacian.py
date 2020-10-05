"""
Created at 20.03.2020
"""

import numba
from ..arakawa_c.indexers import indexers
from ..arakawa_c.enumerations import MAX_DIM_NUM
from ..arakawa_c.traversals import Traversals
from ..options import Options


def make_laplacian(non_unit_g_factor: bool, options: Options, traversals: Traversals):
    if not options.non_zero_mu_coeff:
        @numba.njit(**options.jit_flags)
        def apply(_1, _2, _3, _4, _5):
            return
    else:
        idx = indexers[traversals.n_dims]
        apply_vector = traversals.apply_vector()

        formulae_laplacian = tuple([
            __make_laplacian(options.jit_flags, idx.at[i], options.epsilon, non_unit_g_factor)
            if idx.at[i] is not None else None
            for i in range(MAX_DIM_NUM)
        ])

        @numba.njit(**options.jit_flags)
        def apply(advector, advectee, advectee_bc, null_vecfield_bc):
            null_vecfield = advector
            null_scalarfield = advectee
            return apply_vector(*formulae_laplacian, *advector, *advectee, *advectee_bc, *null_vecfield, *null_vecfield_bc,
                                *null_scalarfield, *null_vecfield_bc)

    return apply


def __make_laplacian(jit_flags, at, epsilon, non_unit_g_factor):
    if non_unit_g_factor:
        raise NotImplementedError()

    @numba.njit(**jit_flags)
    def A(advectee, _, __):
        return -2 * (
            at(*advectee, 1, 0, 0) - at(*advectee, 0, 0, 0)
        ) / (
            at(*advectee, 1, 0, 0) + at(*advectee, 0, 0, 0) + epsilon
        )
    return A

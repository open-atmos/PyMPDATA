import numpy as np
import numba
from ..arakawa_c.indexers import indexers
from ..arakawa_c.enumerations import MAX_DIM_NUM, INNER, OUTER
# TODO #133: rename file


def make_psi_extremum(extremum, options, traversals):
    if not options.flux_corrected_transport:
        @numba.njit(**options.jit_flags)
        def apply(_psi_extremum, _psi, _psi_bc, _null_vecfield, _null_vecfield_bc):
            return
    else:
        idx = indexers[traversals.n_dims]
        apply_scalar = traversals.apply_scalar(loop=False)

        at_idx = INNER if traversals.n_dims == 1 else OUTER
        formulae = (__make_psi_extremum(options.jit_flags, traversals.n_dims, idx.at[at_idx], extremum), None, None)

        null_scalfield, null_scalfield_bc = traversals.null_scalar_field.impl

        @numba.njit(**options.jit_flags)
        def apply(psi_extremum, psi, psi_bc, null_vecfield, null_vecfield_bc):
            return apply_scalar(*formulae, *psi_extremum, *null_vecfield, *null_vecfield_bc, *psi, *psi_bc,
                                *null_scalfield, *null_scalfield_bc, *null_scalfield, *null_scalfield_bc)

    return apply


def __make_psi_extremum(jit_flags, n_dims, at, extremum):
    if n_dims == 1:
        @numba.njit(**jit_flags)
        def psi_extremum(_0, _1, psi, _3, _4):
            return extremum(at(*psi, 0), at(*psi, -1), at(*psi, 1))
    elif n_dims == 2:
        @numba.njit(**jit_flags)
        def psi_extremum(_0, _1, psi, _3, _4):
            return extremum(at(*psi, 0, 0), at(*psi, -1, 0), at(*psi, 1, 0), at(*psi, 0, -1), at(*psi, 0, 1))
    elif n_dims == 3:
        def psi_extremum(_0, _1, psi, _3, _4):
            return extremum(
                at(*psi, 0, 0, 0),
                at(*psi, -1, 0, 0), at(*psi, 1, 0, 0),
                at(*psi, 0, -1, 0), at(*psi, 0, 1, 0),
                at(*psi, 0, 0, -1), at(*psi, 0, 0, 1),
            )
    else:
        raise NotImplementedError()
    return psi_extremum


def make_beta(extremum, non_unit_g_factor, options, traversals):
    if not options.flux_corrected_transport:
        @numba.njit(**options.jit_flags)
        def apply(_beta, _flux, _flux_bc, _psi, _psi_bc, _psi_extremum, _psi_extremum_bc,
                  _g_factor, _g_factor_bc):
            return
    else:
        idx = indexers[traversals.n_dims]
        apply_scalar = traversals.apply_scalar(loop=False)
        at_idx = INNER if traversals.n_dims == 1 else OUTER
        formulae = (
            __make_beta(
                options.jit_flags, traversals.n_dims, idx.at[at_idx], idx.atv[at_idx],
                non_unit_g_factor, options.epsilon, extremum
            ),
            None,
            None
        )

        @numba.njit(**options.jit_flags)
        def apply(beta, flux, flux_bc, psi, psi_bc, psi_extremum, psi_extremum_bc, g_factor, g_factor_bc):
            return apply_scalar(*formulae, *beta, *flux, *flux_bc, *psi, *psi_bc,
                                *psi_extremum, *psi_extremum_bc, *g_factor, *g_factor_bc)

    return apply


def __make_beta(jit_flags, n_dims, at, atv, non_unit_g_factor, epsilon, extremum):
    sign = -1 if extremum == min else 1
    if n_dims == 1:
        @numba.njit(**jit_flags)
        def denominator(flux):
            return max(atv(*flux, -.5 * sign), 0) - min(atv(*flux, +.5 * sign), 0) + epsilon
    elif n_dims == 2:
        @numba.njit(**jit_flags)
        def denominator(flux):
            return (
                max(atv(*flux, -.5 * sign, 0), 0) - min(atv(*flux, +.5 * sign, 0), 0) +
                max(atv(*flux, 0, -.5 * sign), 0) - min(atv(*flux, 0, +.5 * sign), 0) +
                epsilon
            )
    elif n_dims == 3:
        @numba.njit(**jit_flags)
        def denominator(flux):
            return (
                max(atv(*flux, -.5 * sign, 0, 0), 0) - min(atv(*flux, +.5 * sign, 0, 0), 0) +
                max(atv(*flux, 0, -.5 * sign, 0), 0) - min(atv(*flux, 0, +.5 * sign, 0), 0) +
                max(atv(*flux, 0, 0, -.5 * sign), 0) - min(atv(*flux, 0, 0, +.5 * sign), 0) +
                epsilon
            )
    else:
        raise NotImplementedError()

    if non_unit_g_factor:
        @numba.njit(**jit_flags)
        def G(g_factor):
            return at(*g_factor, 0)
    else:
        @numba.njit(**jit_flags)
        def G(_):
            return 1

    if n_dims == 1:
        @numba.njit(**jit_flags)
        def psi_extremum(_0, flux, psi, psi_ext, g_factor):
            return ((extremum(at(*psi_ext, 0), at(*psi, 0), at(*psi, -1), at(*psi, 1))
                     - at(*psi, 0)) * sign * G(g_factor)
                    ) / denominator(flux)

    elif n_dims == 2:
        @numba.njit(**jit_flags)
        def psi_extremum(_0, flux, psi, psi_ext, g_factor):
            return ((extremum(at(*psi_ext, 0, 0),
                              at(*psi, 0, 0),
                              at(*psi, -1, 0), at(*psi, 1, 0),
                              at(*psi, 0, -1), at(*psi, 0, 1))
                     - at(*psi, 0, 0)) * sign * G(g_factor)
                    ) / denominator(flux)
    elif n_dims == 3:
        @numba.njit(**jit_flags)
        def psi_extremum(_0, flux, psi, psi_ext, g_factor):
            return ((extremum(at(*psi_ext, 0, 0, 0),
                              at(*psi, 0, 0, 0),
                              at(*psi, -1, 0, 0), at(*psi, 1, 0, 0),
                              at(*psi, 0, -1, 0), at(*psi, 0, 1, 0),
                              at(*psi, 0, 0, -1), at(*psi, 0, 0, 1))
                     - at(*psi, 0, 0, 0)) * sign * G(g_factor)
                    ) / denominator(flux)
    else:
        raise NotImplementedError()
    return psi_extremum


def make_correction(options, traversals):
    if not options.flux_corrected_transport:
        @numba.njit(**options.jit_flags)
        def apply(_GC_corr, _vec_bc, _beta_down, _beta_down_bc, _beta_up, _beta_up_bc):
            return
    else:
        idx = indexers[traversals.n_dims]
        apply_vector = traversals.apply_vector()

        formulae = tuple([
            __make_correction(options.jit_flags, idx.at[i], idx.atv[i])
            if idx.at[i] is not None else None
            for i in range(MAX_DIM_NUM)
        ])

        @numba.njit(**options.jit_flags)
        def apply(GC_corr, vec_bc, beta_down, beta_down_bc, beta_up, beta_up_bc):
            return apply_vector(*formulae, *GC_corr, *beta_down, *beta_down_bc, *GC_corr, *vec_bc,
                                *beta_up, *beta_up_bc)

    return apply


def __make_correction(jit_flags, at, atv):
    @numba.njit(**jit_flags)
    def correction(beta_down, GC, beta_up):
        a = min(1, at(*beta_down, 0), at(*beta_up, 1))
        b = min(1, at(*beta_up, 0), at(*beta_down, 1))
        c = atv(*GC, +.5)
        return (c + np.abs(c)) / 2 * a + (c - np.abs(c)) / 2 * b

    return correction

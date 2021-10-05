import numpy as np
import numba
from ..arakawa_c.indexers import indexers
from ..arakawa_c.enumerations import MAX_DIM_NUM, INNER, OUTER, META_AND_DATA_META, META_AND_DATA_DATA
# TODO #133: rename file


def make_psi_extrema(options, traversals):
    if not options.flux_corrected_transport:
        @numba.njit(**options.jit_flags)
        def apply(_psi_extrema, _psi, _psi_bc, _null_vecfield, _null_vecfield_bc):
            return
    else:
        idx = indexers[traversals.n_dims]
        apply_scalar = traversals.apply_scalar(loop=False)

        at_idx = INNER if traversals.n_dims == 1 else OUTER
        formulae = (__make_psi_extrema(options.jit_flags, traversals.n_dims, idx.at[at_idx]), None, None)

        null_scalfield, null_scalfield_bc = traversals.null_scalar_field.impl

        @numba.njit(**options.jit_flags)
        def apply(psi_extrema, psi, psi_bc, null_vecfield, null_vecfield_bc):
            return apply_scalar(*formulae,
                                *psi_extrema,
                                *null_vecfield, *null_vecfield_bc,
                                *psi, *psi_bc,
                                *null_scalfield, *null_scalfield_bc,
                                *null_scalfield, *null_scalfield_bc,
                                *null_scalfield, *null_scalfield_bc
                                )

    return apply


def __make_psi_extrema(jit_flags, n_dims, at):
    if n_dims == 1:
        @numba.njit(**jit_flags)
        def _impl(psi, extremum):
            return extremum(
                at(*psi, 0),
                at(*psi, -1), at(*psi, 1)
            )
    elif n_dims == 2:
        @numba.njit(**jit_flags)
        def _impl(psi, extremum):
            return extremum(
                at(*psi, 0, 0),
                at(*psi, -1, 0), at(*psi, 1, 0),
                at(*psi, 0, -1), at(*psi, 0, 1)
            )
    elif n_dims == 3:
        @numba.njit(**jit_flags)
        def _impl(psi, extremum):
            return extremum(
                at(*psi, 0, 0, 0),
                at(*psi, -1, 0, 0), at(*psi, 1, 0, 0),
                at(*psi, 0, -1, 0), at(*psi, 0, 1, 0),
                at(*psi, 0, 0, -1), at(*psi, 0, 0, 1),
            )
    else:
        raise NotImplementedError()

    @numba.njit(**jit_flags)
    def psi_extremum(_0, _1, psi, _3, _4, _5):
        return complex(
            _impl(psi, min),
            _impl(psi, max)
        )

    return psi_extremum


def make_beta(non_unit_g_factor, options, traversals):
    if not options.flux_corrected_transport:
        @numba.njit(**options.jit_flags)
        def apply(_beta, _flux, _flux_bc, _psi, _psi_bc, _psi_extrema, _psi_extrema_bc,
                  _g_factor, _g_factor_bc):
            return
    else:
        idx = indexers[traversals.n_dims]
        apply_scalar = traversals.apply_scalar(loop=False)
        at_idx = INNER if traversals.n_dims == 1 else OUTER
        formulae = (
            __make_beta(
                options.jit_flags, traversals.n_dims, idx.at[at_idx], idx.atv[at_idx],
                non_unit_g_factor, options.epsilon
            ),
            None,
            None
        )

        null_scalfield, null_scalfield_bc = traversals.null_scalar_field.impl

        @numba.njit(**options.jit_flags)
        def apply(beta, flux, flux_bc, psi, psi_bc, psi_extrema, psi_extrema_bc, g_factor, g_factor_bc):
            return apply_scalar(*formulae,
                                *beta,
                                *flux, *flux_bc,
                                *psi, *psi_bc,
                                *psi_extrema, *psi_extrema_bc,
                                *g_factor, *g_factor_bc,
                                *null_scalfield, *null_scalfield_bc
                                )

    return apply


def __make_beta(jit_flags, n_dims, at, atv, non_unit_g_factor, epsilon):
    if n_dims == 1:
        @numba.njit(**jit_flags)
        def denominator(flux, sign):
            return max(atv(*flux, -.5 * sign), 0) - min(atv(*flux, +.5 * sign), 0) + epsilon
    elif n_dims == 2:
        @numba.njit(**jit_flags)
        def denominator(flux, sign):
            return (
                max(atv(*flux, -.5 * sign, 0), 0) - min(atv(*flux, +.5 * sign, 0), 0) +
                max(atv(*flux, 0, -.5 * sign), 0) - min(atv(*flux, 0, +.5 * sign), 0) +
                epsilon
            )
    elif n_dims == 3:
        @numba.njit(**jit_flags)
        def denominator(flux, sign):
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
        def _impl(flux, psi, psi_ext, g_factor, extremum, sign):
            return ((extremum(at(*psi_ext, 0), at(*psi, 0), at(*psi, -1), at(*psi, 1))
                     - at(*psi, 0)) * sign * G(g_factor)
                    ) / denominator(flux, sign)
    elif n_dims == 2:
        @numba.njit(**jit_flags)
        def _impl(flux, psi, psi_ext, g_factor, extremum, sign):
            return ((extremum(at(*psi_ext, 0, 0),
                              at(*psi, 0, 0),
                              at(*psi, -1, 0), at(*psi, 1, 0),
                              at(*psi, 0, -1), at(*psi, 0, 1))
                     - at(*psi, 0, 0)) * sign * G(g_factor)
                    ) / denominator(flux, sign)
    elif n_dims == 3:
        @numba.njit(**jit_flags)
        def _impl(flux, psi, psi_ext, g_factor, extremum, sign):
            return ((extremum(at(*psi_ext, 0, 0, 0),
                              at(*psi, 0, 0, 0),
                              at(*psi, -1, 0, 0), at(*psi, 1, 0, 0),
                              at(*psi, 0, -1, 0), at(*psi, 0, 1, 0),
                              at(*psi, 0, 0, -1), at(*psi, 0, 0, 1))
                     - at(*psi, 0, 0, 0)) * sign * G(g_factor)
                    ) / denominator(flux, sign)
    else:
        raise NotImplementedError()

    @numba.njit(**jit_flags)
    def psi_extremum(_0, flux, psi, psi_ext, g_factor, _5):
        psi_min = (psi_ext[META_AND_DATA_META], psi_ext[META_AND_DATA_DATA].real)
        psi_max = (psi_ext[META_AND_DATA_META], psi_ext[META_AND_DATA_DATA].imag)
        return complex(
            _impl(flux, psi, psi_min, g_factor, min, -1),
            _impl(flux, psi, psi_max, g_factor, max, +1)
        )
    return psi_extremum


def make_correction(options, traversals):
    if not options.flux_corrected_transport:
        @numba.njit(**options.jit_flags)
        def apply(_GC_corr, _vec_bc, _beta, _beta_bc):
            return
    else:
        idx = indexers[traversals.n_dims]
        apply_vector = traversals.apply_vector()

        formulae = tuple([
            __make_correction(options.jit_flags, idx.at[i], idx.atv[i])
            if idx.at[i] is not None else None
            for i in range(MAX_DIM_NUM)
        ])

        null_scalfield, null_scalfield_bc = traversals.null_scalar_field.impl

        @numba.njit(**options.jit_flags)
        def apply(GC_corr, vec_bc, beta, beta_bc):
            return apply_vector(*formulae,
                                *GC_corr,
                                *beta, *beta_bc,
                                *GC_corr, *vec_bc,
                                *null_scalfield, *null_scalfield_bc
                                )

    return apply


def __make_correction(jit_flags, at, atv):
    @numba.njit(**jit_flags)
    def correction(beta, GC, _):
        beta_down = (beta[META_AND_DATA_META], beta[META_AND_DATA_DATA].real)
        beta_up = (beta[META_AND_DATA_META], beta[META_AND_DATA_DATA].imag)
        a = min(1, at(*beta_down, 0), at(*beta_up, 1))
        b = min(1, at(*beta_up, 0), at(*beta_down, 1))
        c = atv(*GC, +.5)
        return (c + np.abs(c)) / 2 * a + (c - np.abs(c)) / 2 * b

    return correction

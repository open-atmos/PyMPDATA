"""the nonoscillatory option for MPDATA as introduced in
[Smolarkiewicz & Grabowski 1990](https://doi.org/10.1016/0021-9991(90)90105-A)"""

import numba
import numpy as np

from PyMPDATA.impl.enumerations import (
    INNER,
    MAX_DIM_NUM,
    META_AND_DATA_DATA,
    META_AND_DATA_META,
    OUTER,
)


def make_psi_extrema(options, traversals):
    """returns an njit-ted function for use with given traversals"""
    if not options.nonoscillatory:

        @numba.njit(**options.jit_flags)
        def apply(_traversal_data, _psi_extrema, _psi):
            return

    else:
        idx = traversals.indexers[traversals.n_dims]
        apply_scalar = traversals.apply_scalar(loop=False)

        at_idx = INNER if traversals.n_dims == 1 else OUTER
        formulae = (
            __make_psi_extrema(options.jit_flags, traversals.n_dims, idx.ats[at_idx]),
            None,
            None,
        )

        @numba.njit(**options.jit_flags)
        def apply(traversals_data, psi_extrema, psi):
            null_scalfield, null_scalfield_bc = traversals_data.null_scalar_field
            null_vecfield, null_vecfield_bc = traversals_data.null_vector_field
            return apply_scalar(
                *formulae,
                *psi_extrema.field,
                *null_vecfield,
                null_vecfield_bc,
                *psi.field,
                psi.bc,
                *null_scalfield,
                null_scalfield_bc,
                *null_scalfield,
                null_scalfield_bc,
                *null_scalfield,
                null_scalfield_bc,
                traversals_data.buffer
            )

    return apply


def __make_psi_extrema(jit_flags, n_dims, ats):
    if n_dims == 1:

        @numba.njit(**jit_flags)
        def _impl(psi, extremum):
            return extremum(ats(*psi, 0), ats(*psi, -1), ats(*psi, 1))

    elif n_dims == 2:

        @numba.njit(**jit_flags)
        def _impl(psi, extremum):
            return extremum(
                ats(*psi, 0, 0),
                ats(*psi, -1, 0),
                ats(*psi, 1, 0),
                ats(*psi, 0, -1),
                ats(*psi, 0, 1),
            )

    elif n_dims == 3:

        @numba.njit(**jit_flags)
        def _impl(psi, extremum):
            return extremum(
                ats(*psi, 0, 0, 0),
                ats(*psi, -1, 0, 0),
                ats(*psi, 1, 0, 0),
                ats(*psi, 0, -1, 0),
                ats(*psi, 0, 1, 0),
                ats(*psi, 0, 0, -1),
                ats(*psi, 0, 0, 1),
            )

    else:
        raise NotImplementedError()

    @numba.njit(**jit_flags)
    def psi_extremum(_0, _1, psi, _3, _4, _5):
        return complex(_impl(psi, min), _impl(psi, max))

    return psi_extremum


def make_beta(non_unit_g_factor, options, traversals):
    """returns njit-ted function for use with given traversals"""
    if not options.nonoscillatory:

        @numba.njit(**options.jit_flags)
        # pylint: disable=too-many-arguments
        def apply(_traversal_data, _beta, _flux, _psi, _psi_extrema, _g_factor):
            return

    else:
        idx = traversals.indexers[traversals.n_dims]
        apply_scalar = traversals.apply_scalar(loop=False)
        at_idx = INNER if traversals.n_dims == 1 else OUTER
        formulae = (
            __make_beta(
                jit_flags=options.jit_flags,
                n_dims=traversals.n_dims,
                ats=idx.ats[at_idx],
                atv=idx.atv[at_idx],
                non_unit_g_factor=non_unit_g_factor,
                epsilon=options.epsilon,
            ),
            None,
            None,
        )

        @numba.njit(**options.jit_flags)
        # pylint: disable=too-many-arguments
        def apply(traversals_data, beta, flux, psi, psi_extrema, g_factor):
            null_scalfield, null_scalfield_bc = traversals_data.null_scalar_field
            return apply_scalar(
                *formulae,
                *beta.field,
                *flux.field,
                flux.bc,
                *psi.field,
                psi.bc,
                *psi_extrema.field,
                psi_extrema.bc,
                *g_factor.field,
                g_factor.bc,
                *null_scalfield,
                null_scalfield_bc,
                traversals_data.buffer
            )

    return apply


def __make_beta(*, jit_flags, n_dims, ats, atv, non_unit_g_factor, epsilon):
    if n_dims == 1:

        @numba.njit(**jit_flags)
        def denominator(flux, sign):
            return (
                max(atv(*flux, -0.5 * sign), 0)
                - min(atv(*flux, +0.5 * sign), 0)
                + epsilon
            )

    elif n_dims == 2:

        @numba.njit(**jit_flags)
        def denominator(flux, sign):
            return (
                max(atv(*flux, -0.5 * sign, 0), 0)
                - min(atv(*flux, +0.5 * sign, 0), 0)
                + max(atv(*flux, 0, -0.5 * sign), 0)
                - min(atv(*flux, 0, +0.5 * sign), 0)
                + epsilon
            )

    elif n_dims == 3:

        @numba.njit(**jit_flags)
        def denominator(flux, sign):
            return (
                max(atv(*flux, -0.5 * sign, 0, 0), 0)
                - min(atv(*flux, +0.5 * sign, 0, 0), 0)
                + max(atv(*flux, 0, -0.5 * sign, 0), 0)
                - min(atv(*flux, 0, +0.5 * sign, 0), 0)
                + max(atv(*flux, 0, 0, -0.5 * sign), 0)
                - min(atv(*flux, 0, 0, +0.5 * sign), 0)
                + epsilon
            )

    else:
        raise NotImplementedError()

    if non_unit_g_factor:

        @numba.njit(**jit_flags)
        def g_fun(arg):
            return ats(*arg, 0)

    else:

        @numba.njit(**jit_flags)
        def g_fun(_):
            return 1

    if n_dims == 1:

        @numba.njit(**jit_flags)
        # pylint: disable=too-many-arguments
        def _impl(flux, psi, psi_ext, g_factor, extremum, sign):
            return (
                (
                    extremum(
                        ats(*psi_ext, 0), ats(*psi, 0), ats(*psi, -1), ats(*psi, 1)
                    )
                    - ats(*psi, 0)
                )
                * sign
                * g_fun(g_factor)
            ) / denominator(flux, sign)

    elif n_dims == 2:

        @numba.njit(**jit_flags)
        # pylint: disable=too-many-arguments
        def _impl(flux, psi, psi_ext, g_factor, extremum, sign):
            return (
                (
                    extremum(
                        ats(*psi_ext, 0, 0),
                        ats(*psi, 0, 0),
                        ats(*psi, -1, 0),
                        ats(*psi, 1, 0),
                        ats(*psi, 0, -1),
                        ats(*psi, 0, 1),
                    )
                    - ats(*psi, 0, 0)
                )
                * sign
                * g_fun(g_factor)
            ) / denominator(flux, sign)

    elif n_dims == 3:

        @numba.njit(**jit_flags)
        # pylint: disable=too-many-arguments
        def _impl(flux, psi, psi_ext, g_factor, extremum, sign):
            return (
                (
                    extremum(
                        ats(*psi_ext, 0, 0, 0),
                        ats(*psi, 0, 0, 0),
                        ats(*psi, -1, 0, 0),
                        ats(*psi, 1, 0, 0),
                        ats(*psi, 0, -1, 0),
                        ats(*psi, 0, 1, 0),
                        ats(*psi, 0, 0, -1),
                        ats(*psi, 0, 0, 1),
                    )
                    - ats(*psi, 0, 0, 0)
                )
                * sign
                * g_fun(g_factor)
            ) / denominator(flux, sign)

    else:
        raise NotImplementedError()

    @numba.njit(**jit_flags)
    # pylint: disable=too-many-arguments
    def psi_extremum(_, flux, psi, psi_ext, g_factor, __):
        psi_min = (psi_ext[META_AND_DATA_META], psi_ext[META_AND_DATA_DATA].real)
        psi_max = (psi_ext[META_AND_DATA_META], psi_ext[META_AND_DATA_DATA].imag)
        return complex(
            _impl(flux, psi, psi_min, g_factor, min, -1),
            _impl(flux, psi, psi_max, g_factor, max, +1),
        )

    return psi_extremum


def make_correction(options, traversals):
    """returns njit-ted function for use with given traversals"""
    if not options.nonoscillatory:

        @numba.njit(**options.jit_flags)
        def apply(_, __, ___):
            return

    else:
        idx = traversals.indexers[traversals.n_dims]
        apply_vector = traversals.apply_vector()

        formulae = tuple(
            (
                __make_correction(options.jit_flags, idx.ats[i], idx.atv[i])
                if idx.ats[i] is not None
                else None
            )
            for i in range(MAX_DIM_NUM)
        )

        @numba.njit(**options.jit_flags)
        def apply(traversals_data, g_c_corr, beta):
            null_scalfield, null_scalfield_bc = traversals_data.null_scalar_field
            return apply_vector(
                *formulae,
                *g_c_corr.field,
                *beta.field,
                beta.bc,
                *g_c_corr.field,
                g_c_corr.bc,
                *null_scalfield,
                null_scalfield_bc,
                traversals_data.buffer
            )

    return apply


def __make_correction(jit_flags, ats, atv):
    @numba.njit(**jit_flags)
    def correction(beta, g_c, _):
        beta_down = (beta[META_AND_DATA_META], beta[META_AND_DATA_DATA].real)
        beta_up = (beta[META_AND_DATA_META], beta[META_AND_DATA_DATA].imag)
        val_1 = min(1, ats(*beta_down, 0), ats(*beta_up, 1))
        val_2 = min(1, ats(*beta_up, 0), ats(*beta_down, 1))
        val_3 = atv(*g_c, +0.5)
        return (val_3 + np.abs(val_3)) / 2 * val_1 + (val_3 - np.abs(val_3)) / 2 * val_2

    return correction

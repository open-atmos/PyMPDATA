""" halo-filling logic for vector field traversals (incl. multi-threading) """
# pylint: disable=too-many-statements,too-many-locals,too-many-lines

import numba

from PyMPDATA.impl.enumerations import (
    INNER,
    INVALID_INDEX,
    MID3D,
    ONE_FOR_STAGGERED_GRID,
    OUTER,
    RNG_START,
    RNG_STOP,
    SIGN_LEFT,
    SIGN_RIGHT,
)
from PyMPDATA.impl.meta import META_HALO_VALID
from PyMPDATA.impl.traversals_common import make_common


def _make_fill_halos_vector(*, jit_flags, halo, n_dims, chunker, spanner, left_first):
    common = make_common(jit_flags, spanner, chunker)
    halos = ((halo - 1, halo, halo), (halo, halo - 1, halo), (halo, halo, halo - 1))
    # pylint:disable=duplicate-code
    kwargs = {
        "jit_flags": jit_flags,
        "halo": halo,
        "n_dims": n_dims,
        "halos": halos,
        "left_first": left_first,
    }

    outer_outer = __make_outer_outer(**kwargs)
    outer_mid3d = __make_outer_mid3d(**kwargs)
    outer_inner = __make_outer_inner(**kwargs)
    mid3d_outer = __make_mid3d_outer(**kwargs)
    mid3d_mid3d = __make_mid3d_mid3d(**kwargs)
    mid3d_inner = __make_mid3d_inner(**kwargs)
    inner_outer = __make_inner_outer(**kwargs)
    inner_inner = __make_inner_inner(**kwargs)
    inner_mid3d = __make_inner_mid3d(**kwargs)

    @numba.njit(**jit_flags)
    # pylint: disable=too-many-arguments
    def boundary_cond_vector(
        thread_id,
        meta,
        comp_outer,
        comp_mid3d,
        comp_inner,
        fun_outer,
        fun_mid3d,
        fun_inner,
    ):
        if meta[META_HALO_VALID]:
            return
        span, rng_outer, last_thread, first_thread = common(meta, thread_id)

        outer_outer(span, comp_outer, fun_outer, first_thread, last_thread)
        outer_mid3d(span, comp_outer, fun_mid3d, rng_outer, last_thread)
        outer_inner(span, comp_outer, fun_inner, rng_outer, last_thread)

        mid3d_outer(span, comp_mid3d, fun_outer, first_thread, last_thread)
        mid3d_mid3d(span, comp_mid3d, fun_mid3d, rng_outer, last_thread)
        mid3d_inner(span, comp_mid3d, fun_inner, rng_outer, last_thread)

        inner_outer(span, comp_inner, fun_outer, first_thread, last_thread)
        inner_mid3d(span, comp_inner, fun_mid3d, rng_outer, last_thread)
        inner_inner(span, comp_inner, fun_inner, last_thread, rng_outer)

    return boundary_cond_vector


def __make_outer_outer(*, jit_flags, halo, n_dims, halos, left_first, **_kwargs):
    @numba.njit(**jit_flags)
    def outer_outer_left(first_thread, _last_thread, span, comp, fun):
        if first_thread:
            i_rng = range(
                halos[OUTER][OUTER] - 1, -1, -1
            )  # note: non-reverse order assumed in Extrapolated
            j_rng = range(0, span[MID3D] + 2 * halo) if n_dims > 2 else (INVALID_INDEX,)
            k_rng = range(0, span[INNER] + 2 * halos[OUTER][INNER])
            fun(i_rng, j_rng, k_rng, comp, span[OUTER] + 1, SIGN_LEFT)

    @numba.njit(**jit_flags)
    def outer_outer_right(_first_thread, last_thread, span, comp, fun):
        if last_thread:
            i_rng = range(
                span[OUTER] + ONE_FOR_STAGGERED_GRID + halos[OUTER][OUTER],
                span[OUTER] + ONE_FOR_STAGGERED_GRID + 2 * halos[OUTER][OUTER],
            )  # note: non-reverse order assumed in Extrapolated
            j_rng = range(0, span[MID3D] + 2 * halo) if n_dims > 2 else (INVALID_INDEX,)
            k_rng = range(0, span[INNER] + 2 * halos[OUTER][INNER])
            fun(i_rng, j_rng, k_rng, comp, span[OUTER] + 1, SIGN_RIGHT)

    if left_first:
        outer_outer_first = outer_outer_left
        outer_outer_last = outer_outer_right
    else:
        outer_outer_first = outer_outer_right
        outer_outer_last = outer_outer_left

    @numba.njit(**jit_flags)
    def outer_outer(span, comp, fun, first_thread, last_thread):
        if n_dims > 1:
            outer_outer_first(first_thread, last_thread, span, comp, fun)
            outer_outer_last(first_thread, last_thread, span, comp, fun)

    return outer_outer


def __make_outer_mid3d(*, jit_flags, halo, n_dims, halos, left_first, **_kwargs):
    @numba.njit(**jit_flags)
    def outer_mid3d_left_jk(span, comp, fun, i):
        j_rng = range(0, halos[OUTER][MID3D])
        k_rng = range(0, span[INNER] + 2 * halo)
        fun(i, j_rng, k_rng, comp, span[MID3D], SIGN_LEFT)

    @numba.njit(**jit_flags)
    def outer_mid3d_right_jk(span, comp, fun, i):
        j_rng = range(
            span[MID3D] + halos[OUTER][MID3D], span[MID3D] + 2 * halos[OUTER][MID3D]
        )
        k_rng = range(0, span[INNER] + 2 * halo)

        fun(i, j_rng, k_rng, comp, span[MID3D], SIGN_RIGHT)

    if left_first:
        outer_mid3d_first_jk = outer_mid3d_left_jk
        outer_mid3d_last_jk = outer_mid3d_right_jk
    else:
        outer_mid3d_first_jk = outer_mid3d_right_jk
        outer_mid3d_last_jk = outer_mid3d_left_jk

    @numba.njit(**jit_flags)
    def outer_mid3d(span, comp, fun, rng_outer, last_thread):
        if n_dims > 2:
            i_rng = range(
                rng_outer[RNG_START],
                rng_outer[RNG_STOP]
                + (
                    (ONE_FOR_STAGGERED_GRID + 2 * halos[OUTER][OUTER])
                    if last_thread
                    else 0
                ),
            )
            outer_mid3d_first_jk(span, comp, fun, i_rng)
            outer_mid3d_last_jk(span, comp, fun, i_rng)

    return outer_mid3d


def __make_outer_inner(*, jit_flags, halo, n_dims, halos, left_first, **_kwargs):
    @numba.njit(**jit_flags)
    def outer_inner_left_k(span, comp, fun, i, j):
        k_rng = range(0, halos[OUTER][INNER])
        fun(i, j, k_rng, comp, span[INNER], SIGN_LEFT)

    @numba.njit(**jit_flags)
    def outer_inner_right_k(span, comp, fun, i, j):
        k_rng = range(
            span[INNER] + halos[OUTER][INNER],
            span[INNER] + 2 * halos[OUTER][INNER],
        )
        fun(i, j, k_rng, comp, span[INNER], SIGN_RIGHT)

    if left_first:
        outer_inner_first_k = outer_inner_left_k
        outer_inner_last_k = outer_inner_right_k
    else:
        outer_inner_first_k = outer_inner_right_k
        outer_inner_last_k = outer_inner_left_k

    @numba.njit(**jit_flags)
    def outer_inner(span, comp, fun, rng_outer, last_thread):
        if n_dims > 1:
            i_rng = range(
                rng_outer[RNG_START],
                rng_outer[RNG_STOP]
                + (
                    (ONE_FOR_STAGGERED_GRID + 2 * halos[OUTER][OUTER])
                    if last_thread
                    else 0
                ),
            )
            j_rng = range(0, span[MID3D] + 2 * halo) if n_dims > 2 else (INVALID_INDEX,)
            outer_inner_first_k(span, comp, fun, i_rng, j_rng)
            outer_inner_last_k(span, comp, fun, i_rng, j_rng)

    return outer_inner


def __make_mid3d_outer(*, jit_flags, halo, n_dims, halos, left_first, **_kwargs):
    @numba.njit(**jit_flags)
    def mid3d_outer_left(first_thread, _last_thread, span, comp, fun):
        if first_thread:
            i_rng = range(0, halos[MID3D][OUTER])
            j_rng = range(
                0, span[MID3D] + ONE_FOR_STAGGERED_GRID + 2 * halos[MID3D][MID3D]
            )
            k_rng = range(0, span[INNER] + 2 * halo)

            fun(i_rng, j_rng, k_rng, comp, span[OUTER], SIGN_LEFT)

    @numba.njit(**jit_flags)
    def mid3d_outer_right(_first_thread, last_thread, span, comp, fun):
        if last_thread:
            i_rng = range(
                span[OUTER] + halos[MID3D][OUTER], span[OUTER] + 2 * halos[MID3D][OUTER]
            )
            j_rng = range(
                0, span[MID3D] + ONE_FOR_STAGGERED_GRID + 2 * halos[MID3D][MID3D]
            )
            k_rng = range(0, span[INNER] + 2 * halo)

            fun(i_rng, j_rng, k_rng, comp, span[OUTER], SIGN_RIGHT)

    if left_first:
        mid3d_outer_first = mid3d_outer_left
        mid3d_outer_last = mid3d_outer_right
    else:
        mid3d_outer_first = mid3d_outer_right
        mid3d_outer_last = mid3d_outer_left

    @numba.njit(**jit_flags)
    def mid3d_outer(span, comp, fun, first_thread, last_thread):
        if n_dims > 2:
            mid3d_outer_first(first_thread, last_thread, span, comp, fun)
            mid3d_outer_last(first_thread, last_thread, span, comp, fun)

    return mid3d_outer


def __make_mid3d_mid3d(*, jit_flags, halo, n_dims, halos, left_first, **_kwargs):
    @numba.njit(**jit_flags)
    def mid3d_mid3d_left_jk(span, comp, fun, i):
        j_rng = range(0, halos[MID3D][MID3D])
        k_rng = range(0, span[INNER] + 2 * halo)

        fun(i, j_rng, k_rng, comp, span[MID3D] + ONE_FOR_STAGGERED_GRID, SIGN_LEFT)

    @numba.njit(**jit_flags)
    def mid3d_mid3d_right_jk(span, comp, fun, i):
        j_rng = range(
            span[MID3D] + 1 + halos[MID3D][MID3D],
            span[MID3D] + ONE_FOR_STAGGERED_GRID + 2 * halos[MID3D][MID3D],
        )
        k_rng = range(0, span[INNER] + 2 * halo)

        fun(i, j_rng, k_rng, comp, span[MID3D] + ONE_FOR_STAGGERED_GRID, SIGN_RIGHT)

    if left_first:
        mid3d_mid3d_first_jk = mid3d_mid3d_left_jk
        mid3d_mid3d_last_jk = mid3d_mid3d_right_jk
    else:
        mid3d_mid3d_first_jk = mid3d_mid3d_right_jk
        mid3d_mid3d_last_jk = mid3d_mid3d_left_jk

    @numba.njit(**jit_flags)
    def mid3d_mid3d(span, comp, fun, rng_outer, last_thread):
        if n_dims > 2:
            i_rng = range(
                rng_outer[RNG_START],
                rng_outer[RNG_STOP] + (2 * halos[MID3D][OUTER] if last_thread else 0),
            )
            mid3d_mid3d_first_jk(span, comp, fun, i_rng)
            mid3d_mid3d_last_jk(span, comp, fun, i_rng)

    return mid3d_mid3d


def __make_mid3d_inner(*, jit_flags, n_dims, halos, left_first, **_kwargs):
    @numba.njit(**jit_flags)
    def mid3d_inner_left_k(span, comp, fun, i, j):
        k_rng = range(0, halos[MID3D][INNER])

        fun(i, j, k_rng, comp, span[INNER], SIGN_LEFT)

    @numba.njit(**jit_flags)
    def mid3d_inner_right_k(span, comp, fun, i, j):
        k_rng = range(
            span[INNER] + halos[MID3D][INNER],
            span[INNER] + 2 * halos[MID3D][INNER],
        )
        fun(i, j, k_rng, comp, span[INNER], SIGN_RIGHT)

    if left_first:
        mid3d_inner_first_k = mid3d_inner_left_k
        mid3d_inner_last_k = mid3d_inner_right_k
    else:
        mid3d_inner_first_k = mid3d_inner_right_k
        mid3d_inner_last_k = mid3d_inner_left_k

    @numba.njit(**jit_flags)
    def mid3d_inner(span, comp, fun, rng_outer, last_thread):
        if n_dims > 2:
            i_rng = range(
                rng_outer[RNG_START],
                rng_outer[RNG_STOP] + (2 * halos[MID3D][OUTER] if last_thread else 0),
            )
            j_rng = range(
                0, span[MID3D] + ONE_FOR_STAGGERED_GRID + 2 * halos[MID3D][MID3D]
            )
            mid3d_inner_first_k(span, comp, fun, i_rng, j_rng)
            mid3d_inner_last_k(span, comp, fun, i_rng, j_rng)

    return mid3d_inner


def __make_inner_outer(*, jit_flags, halo, n_dims, halos, left_first, **_kwargs):
    @numba.njit(**jit_flags)
    def inner_outer_left(first_thread, _last_thread, span, comp, fun):
        if first_thread:
            i_rng = range(0, halos[INNER][OUTER])
            j_rng = range(0, span[MID3D] + 2 * halo) if n_dims > 2 else (INVALID_INDEX,)
            k_rng = range(
                0,
                span[INNER] + ONE_FOR_STAGGERED_GRID + 2 * halos[INNER][INNER],
            )

            fun(i_rng, j_rng, k_rng, comp, span[OUTER], SIGN_LEFT)

    @numba.njit(**jit_flags)
    def inner_outer_right(_first_thread, last_thread, span, comp, fun):
        if last_thread:
            i_rng = range(
                span[OUTER] + halos[INNER][OUTER], span[OUTER] + 2 * halos[INNER][OUTER]
            )
            j_rng = range(0, span[MID3D] + 2 * halo) if n_dims > 2 else (INVALID_INDEX,)
            k_rng = range(
                0,
                span[INNER] + ONE_FOR_STAGGERED_GRID + 2 * halos[INNER][INNER],
            )
            fun(i_rng, j_rng, k_rng, comp, span[OUTER], SIGN_RIGHT)

    if left_first:
        inner_outer_first = inner_outer_left
        inner_outer_last = inner_outer_right
    else:
        inner_outer_first = inner_outer_right
        inner_outer_last = inner_outer_left

    @numba.njit(**jit_flags)
    def inner_outer(span, comp, fun, first_thread, last_thread):
        if n_dims > 1:
            inner_outer_first(first_thread, last_thread, span, comp, fun)
            inner_outer_last(first_thread, last_thread, span, comp, fun)

    return inner_outer


def __make_inner_inner(*, jit_flags, halo, n_dims, halos, left_first, **_kwargs):
    @numba.njit(**jit_flags)
    def inner_inner_left_k(span, comp, fun, i, j):
        k_rng = range(0, halos[INNER][INNER])
        fun(
            i,
            j,
            k_rng,
            comp,
            span[INNER] + ONE_FOR_STAGGERED_GRID,
            SIGN_LEFT,
        )

    @numba.njit(**jit_flags)
    def inner_inner_right_k(span, comp, fun, i, j):
        k_rng = range(
            span[INNER] + 1 + halos[INNER][INNER],
            span[INNER] + ONE_FOR_STAGGERED_GRID + 2 * halos[INNER][INNER],
        )
        fun(
            i,
            j,
            k_rng,
            comp,
            span[INNER] + ONE_FOR_STAGGERED_GRID,
            SIGN_RIGHT,
        )

    if left_first:
        inner_inner_first_k = inner_inner_left_k
        inner_inner_last_k = inner_inner_right_k
    else:
        inner_inner_first_k = inner_inner_right_k
        inner_inner_last_k = inner_inner_left_k

    @numba.njit(**jit_flags)
    def inner_inner(span, comp, fun, last_thread, rng_outer):
        i_rng = (
            range(
                rng_outer[RNG_START],
                rng_outer[RNG_STOP] + (2 * halos[INNER][OUTER] if last_thread else 0),
            )
            if n_dims > 1
            else (INVALID_INDEX,)
        )
        j_rng = range(0, span[MID3D] + 2 * halo) if n_dims > 2 else (INVALID_INDEX,)

        inner_inner_first_k(span, comp, fun, i_rng, j_rng)
        inner_inner_last_k(span, comp, fun, i_rng, j_rng)

    return inner_inner


def __make_inner_mid3d(*, jit_flags, n_dims, halos, left_first, **_kwargs):
    @numba.njit(**jit_flags)
    def inner_mid3d_left_jk(span, comp, fun, i):
        j_rng = range(0, halos[INNER][MID3D])
        k_rng = range(0, span[INNER] + ONE_FOR_STAGGERED_GRID + 2 * halos[INNER][INNER])
        fun(i, j_rng, k_rng, comp, span[MID3D], SIGN_LEFT)

    @numba.njit(**jit_flags)
    def inner_mid3d_right_jk(span, comp, fun, i):
        j_rng = range(
            span[MID3D] + halos[INNER][MID3D], span[MID3D] + 2 * halos[INNER][MID3D]
        )
        k_rng = range(0, span[INNER] + ONE_FOR_STAGGERED_GRID + 2 * halos[INNER][INNER])
        fun(i, j_rng, k_rng, comp, span[MID3D], SIGN_RIGHT)

    if left_first:
        inner_mid3d_first_jk = inner_mid3d_left_jk
        inner_mid3d_last_jk = inner_mid3d_right_jk
    else:
        inner_mid3d_first_jk = inner_mid3d_right_jk
        inner_mid3d_last_jk = inner_mid3d_left_jk

    @numba.njit(**jit_flags)
    def inner_mid3d(span, comp, fun, rng_outer, last_thread):
        if n_dims > 2:
            i_rng = range(
                rng_outer[RNG_START],
                rng_outer[RNG_STOP] + (2 * halos[INNER][OUTER] if last_thread else 0),
            )
            inner_mid3d_first_jk(span, comp, fun, i_rng)
            inner_mid3d_last_jk(span, comp, fun, i_rng)

    return inner_mid3d

""" halo-filling logic for scalar field traversals (incl. multi-threading) """
import numba

from PyMPDATA.impl.enumerations import (
    INNER,
    INVALID_INDEX,
    MID3D,
    OUTER,
    RNG_START,
    RNG_STOP,
    SIGN_LEFT,
    SIGN_RIGHT,
)
from PyMPDATA.impl.meta import META_HALO_VALID
from PyMPDATA.impl.traversals_common import _make_common


def _make_fill_halos_scalar(*, jit_flags, halo, n_dims, chunker, spanner, left_first):
    common = _make_common(jit_flags, spanner, chunker)
    kwargs = {
        "jit_flags": jit_flags,
        "halo": halo,
        "n_dims": n_dims,
        "left_first": left_first,
    }
    mid3d = __make_mid3d(**kwargs)
    outer = __make_outer(**kwargs)
    inner = __make_inner(**kwargs)

    @numba.njit(**jit_flags)
    # pylint: disable=too-many-arguments,too-many-branches
    def boundary_cond_scalar(thread_id, meta, psi, fun_outer, fun_mid3d, fun_inner):
        if meta[META_HALO_VALID]:
            return
        span, rng_outer, last_thread, first_thread = common(meta, thread_id)
        mid3d(last_thread, rng_outer, span, psi, fun_mid3d)
        outer(first_thread, last_thread, span, psi, fun_outer)
        inner(last_thread, rng_outer, span, psi, fun_inner)

    return boundary_cond_scalar


def __make_mid3d(*, jit_flags, halo, n_dims, left_first):
    @numba.njit(**jit_flags)
    def mid3d_right_j(span, psi, fun_mid3d, i_rng, k_rng):
        j_rng = range(span[MID3D] + halo, span[MID3D] + 2 * halo)
        fun_mid3d(i_rng, j_rng, k_rng, psi, span[MID3D], SIGN_RIGHT)

    @numba.njit(**jit_flags)
    def mid3d_left_j(span, psi, fun_mid3d, i_rng, k_rng):
        j_rng = range(halo - 1, -1, -1)  # note: reversed order for Extrapolated!
        fun_mid3d(i_rng, j_rng, k_rng, psi, span[MID3D], SIGN_LEFT)

    if left_first:
        mid3d_first_j = mid3d_left_j
        mid3d_last_j = mid3d_right_j
    else:
        mid3d_first_j = mid3d_right_j
        mid3d_last_j = mid3d_left_j

    @numba.njit(**jit_flags)
    def mid3d(last_thread, rng_outer, span, psi, fun_mid3d):
        if n_dims > 2:
            i_rng = range(
                rng_outer[RNG_START],
                rng_outer[RNG_STOP] + (2 * halo if last_thread else 0),
            )
            k_rng = range(0, span[INNER] + 2 * halo)
            mid3d_first_j(span, psi, fun_mid3d, i_rng, k_rng)
            mid3d_last_j(span, psi, fun_mid3d, i_rng, k_rng)

    return mid3d


def __make_outer(*, jit_flags, halo, n_dims, left_first):
    @numba.njit(**jit_flags)
    def outer_left(first_thread, _last_thread, span, psi, fun_outer):
        if first_thread:
            i_rng = range(
                halo - 1, -1, -1
            )  # note: reversed order assumed in Extrapolated
            j_rng = range(0, span[MID3D] + 2 * halo) if n_dims > 2 else (INVALID_INDEX,)
            k_rng = range(0, span[INNER] + 2 * halo)
            fun_outer(i_rng, j_rng, k_rng, psi, span[OUTER], SIGN_LEFT)

    @numba.njit(**jit_flags)
    def outer_right(_first_thread, last_thread, span, psi, fun_outer):
        if last_thread:
            i_rng = range(
                span[OUTER] + halo, span[OUTER] + 2 * halo
            )  # note: non-reversed order for Extrapolated
            j_rng = range(0, span[MID3D] + 2 * halo) if n_dims > 2 else (INVALID_INDEX,)
            k_rng = range(0, span[INNER] + 2 * halo)
            fun_outer(i_rng, j_rng, k_rng, psi, span[OUTER], SIGN_RIGHT)

    if left_first:
        outer_first_ijk = outer_left
        outer_last_ijk = outer_right
    else:
        outer_first_ijk = outer_right
        outer_last_ijk = outer_left

    @numba.njit(**jit_flags)
    def outer(first_thread, last_thread, span, psi, fun_outer):
        if n_dims > 1:
            outer_first_ijk(first_thread, last_thread, span, psi, fun_outer)
            outer_last_ijk(first_thread, last_thread, span, psi, fun_outer)

    return outer


def __make_inner(*, jit_flags, halo, n_dims, left_first):
    @numba.njit(**jit_flags)
    def inner_left_k(span, psi, fun_inner, i_rng, j_rng):
        k_rng = range(halo - 1, -1, -1)  # note: reversed order assumed in Extrapolated!
        fun_inner(i_rng, j_rng, k_rng, psi, span[INNER], SIGN_LEFT)

    @numba.njit(**jit_flags)
    def inner_right_k(span, psi, fun_inner, i_rng, j_rng):
        k_rng = range(span[INNER] + halo, span[INNER] + 2 * halo)
        fun_inner(i_rng, j_rng, k_rng, psi, span[INNER], SIGN_RIGHT)

    if left_first:
        inner_first_k = inner_left_k
        inner_last_k = inner_right_k
    else:
        inner_first_k = inner_right_k
        inner_last_k = inner_left_k

    @numba.njit(**jit_flags)
    def inner(last_thread, rng_outer, span, psi, fun_inner):
        i_rng = (
            range(
                rng_outer[RNG_START],
                rng_outer[RNG_STOP] + (2 * halo if last_thread else 0),
            )
            if n_dims > 1
            else (INVALID_INDEX,)
        )
        j_rng = range(0, span[MID3D] + 2 * halo) if n_dims > 2 else (INVALID_INDEX,)

        inner_first_k(span, psi, fun_inner, i_rng, j_rng)
        inner_last_k(span, psi, fun_inner, i_rng, j_rng)

    return inner

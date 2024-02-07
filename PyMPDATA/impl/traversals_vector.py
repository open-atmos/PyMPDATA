""" vector field traversals (incl. multi-threading) """

import numba

from .enumerations import (
    INNER,
    INVALID_INDEX,
    MID3D,
    ONE_FOR_STAGGERED_GRID,
    OUTER,
    RNG_START,
    RNG_STOP,
)
from .meta import META_HALO_VALID
from .traversals_common import make_common


def _make_apply_vector(
    *,
    indexers,
    jit_flags,
    halo,
    n_dims,
    n_threads,
    spanner,
    chunker,
    boundary_cond_vector,
    boundary_cond_scalar
):
    set_value = indexers[n_dims].set
    common = make_common(jit_flags, spanner, chunker)
    halos = ((halo - 1, halo, halo), (halo, halo - 1, halo), (halo, halo, halo - 1))

    @numba.njit(**jit_flags)
    # pylint: disable=too-many-arguments,too-many-locals
    def apply_vector_impl(
        thread_id,
        out_meta,
        fun_outer,
        fun_mid3d,
        fun_inner,
        out_outer,
        out_mid3d,
        out_inner,
        scal_arg1,
        vec_arg2_outer,
        vec_arg2_mid3d,
        vec_arg2_inner,
        scal_arg3,
    ):
        span, rng_outer, last_thread, _ = common(out_meta, thread_id)
        rng_mid3d = (0, span[MID3D])
        rng_inner = (0, span[INNER])
        arg2 = (vec_arg2_outer, vec_arg2_mid3d, vec_arg2_inner)

        for i in (
            range(
                rng_outer[RNG_START] + halos[OUTER][OUTER],
                rng_outer[RNG_STOP]
                + halos[OUTER][OUTER]
                + (ONE_FOR_STAGGERED_GRID if last_thread else 0),
            )
            if n_dims > 1
            else (INVALID_INDEX,)
        ):
            for j in (
                range(
                    rng_mid3d[RNG_START] + halos[MID3D][MID3D],
                    rng_mid3d[RNG_STOP] + ONE_FOR_STAGGERED_GRID + halos[MID3D][MID3D],
                )
                if n_dims > 2
                else (INVALID_INDEX,)
            ):
                for k in range(
                    rng_inner[RNG_START] + halos[INNER][INNER],
                    rng_inner[RNG_STOP] + ONE_FOR_STAGGERED_GRID + halos[INNER][INNER],
                ):
                    focus = (i, j, k)
                    if n_dims > 1:
                        set_value(
                            out_outer,
                            i,
                            j,
                            k,
                            fun_outer(
                                (focus, scal_arg1), (focus, arg2), (focus, scal_arg3)
                            ),
                        )
                        if n_dims > 2:
                            set_value(
                                out_mid3d,
                                i,
                                j,
                                k,
                                fun_mid3d(
                                    (focus, scal_arg1),
                                    (focus, arg2),
                                    (focus, scal_arg3),
                                ),
                            )
                    set_value(
                        out_inner,
                        i,
                        j,
                        k,
                        fun_inner(
                            (focus, scal_arg1), (focus, arg2), (focus, scal_arg3)
                        ),
                    )

    @numba.njit(**{**jit_flags, **{"parallel": n_threads > 1}})
    # pylint: disable=too-many-arguments,too-many-locals
    def apply_vector(
        fun_outer,
        fun_mid3d,
        fun_inner,
        out_meta,
        out_outer,
        out_mid3d,
        out_inner,
        arg1s_meta,
        arg1s_data,
        arg1s_bc,
        arg2v_meta,
        arg2v_data_o,
        arg2v_data_m,
        arg2v_data_i,
        arg2v_bc,
        arg3s_meta,
        arg3s_data,
        arg3s_bc,
        buffer,
    ):
        for thread_id in range(1) if n_threads == 1 else numba.prange(n_threads):
            boundary_cond_scalar(thread_id, arg1s_meta, arg1s_data, arg1s_bc, buffer)
            boundary_cond_vector(
                thread_id,
                arg2v_meta,
                (arg2v_data_o, arg2v_data_m, arg2v_data_i),
                arg2v_bc,
                buffer,
            )
            boundary_cond_scalar(thread_id, arg3s_meta, arg3s_data, arg3s_bc, buffer)
        if not arg1s_meta[META_HALO_VALID]:
            arg1s_meta[META_HALO_VALID] = True
        if not arg2v_meta[META_HALO_VALID]:
            arg2v_meta[META_HALO_VALID] = True
        if not arg3s_meta[META_HALO_VALID]:
            arg3s_meta[META_HALO_VALID] = True

        for thread_id in range(1) if n_threads == 1 else numba.prange(n_threads):
            # pylint: disable=duplicate-code
            apply_vector_impl(
                thread_id,
                out_meta,
                fun_outer,
                fun_mid3d,
                fun_inner,
                out_outer,
                out_mid3d,
                out_inner,
                arg1s_data,
                arg2v_data_o,
                arg2v_data_m,
                arg2v_data_i,
                arg3s_data,
            )
        out_meta[META_HALO_VALID] = False

    return apply_vector

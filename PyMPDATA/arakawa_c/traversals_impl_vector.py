import numba

from .indexers import indexers
from .meta import META_HALO_VALID
from .enumerations import OUTER, MID3D, INNER, SIGN_LEFT, SIGN_RIGHT, RNG_STOP, RNG_START, INVALID_INDEX, ONE_FOR_STAGGERED_GRID


def _make_apply_vector(*, jit_flags, halo, n_dims, n_threads, spanner, chunker, boundary_cond_vector,
                       boundary_cond_scalar):
    set = indexers[n_dims].set

    halos = (
        (halo - 1, halo, halo),
        (halo, halo - 1, halo),
        (halo, halo, halo - 1)
    )

    @numba.njit(**jit_flags)
    def apply_vector_impl(thread_id, out_meta,
                          fun_outer, fun_mid3d, fun_inner,
                          out_outer, out_mid3d, out_inner,
                          scal_arg1,
                          vec_arg2_outer, vec_arg2_mid3d, vec_arg2_inner,
                          scal_arg3
                          ):
        span = spanner(out_meta)
        rng_outer = chunker(out_meta, thread_id)
        rng_mid3d = (0, span[MID3D])
        rng_inner = (0, span[INNER])
        last_thread = rng_outer[RNG_STOP] == span[OUTER]
        arg2 = (vec_arg2_outer, vec_arg2_mid3d, vec_arg2_inner)

        for i in range(rng_outer[RNG_START] + halos[OUTER][OUTER], rng_outer[RNG_STOP] + halos[OUTER][OUTER] + (ONE_FOR_STAGGERED_GRID if last_thread else 0)) if n_dims > 1 else (INVALID_INDEX,):
            for j in (INVALID_INDEX,):  # TODO #96
                for k in range(rng_inner[RNG_START] + halos[INNER][INNER], rng_inner[RNG_STOP] + ONE_FOR_STAGGERED_GRID + halos[INNER][INNER]):
                    focus = (i, j, k)
                    if n_dims > 1:
                        set(out_outer, i, j, k, fun_outer((focus, scal_arg1), (focus, arg2), (focus, scal_arg3)))
                        if n_dims > 2:
                            set(out_mid3d, i, j, k, fun_mid3d((focus, scal_arg1), (focus, arg2), (focus, scal_arg3)))
                    set(out_inner, i, j, k, fun_inner((focus, scal_arg1), (focus, arg2), (focus, scal_arg3)))

    @numba.njit(**{**jit_flags, **{'parallel': n_threads > 1}})
    def apply_vector(
            fun_outer, fun_mid3d, fun_inner,
            out_meta, out_outer, out_mid3d, out_inner,
            scal_arg1_meta, scal_arg1, scal_arg1_bc_outer, scal_arg1_bc_mid3d, scal_arg1_bc_inner,
            vec_arg2_meta, vec_arg2_outer, vec_arg2_mid3d, vec_arg2_inner, vec_arg2_bc_outer, vec_arg2_bc_mid3d, vec_arg2_bc_inner,
            scal_arg3_meta, scal_arg3, scal_arg3_bc_outer, scal_arg3_bc_mid3d, scal_arg3_bc_inner
    ):
        for thread_id in range(1) if n_threads == 1 else numba.prange(n_threads):
            boundary_cond_scalar(thread_id, scal_arg1_meta, scal_arg1, scal_arg1_bc_outer, scal_arg1_bc_mid3d, scal_arg1_bc_inner)
            boundary_cond_vector(thread_id, vec_arg2_meta, vec_arg2_outer, vec_arg2_mid3d, vec_arg2_inner, vec_arg2_bc_outer, vec_arg2_bc_mid3d, vec_arg2_bc_inner)
            boundary_cond_scalar(thread_id, scal_arg3_meta, scal_arg3, scal_arg3_bc_outer, scal_arg3_bc_mid3d, scal_arg3_bc_inner)
        if not scal_arg1_meta[META_HALO_VALID]:
            scal_arg1_meta[META_HALO_VALID] = True
        if not vec_arg2_meta[META_HALO_VALID]:
            vec_arg2_meta[META_HALO_VALID] = True
        if not scal_arg3_meta[META_HALO_VALID]:
            scal_arg3_meta[META_HALO_VALID] = True

        for thread_id in range(1) if n_threads == 1 else numba.prange(n_threads):
            apply_vector_impl(
                thread_id,
                out_meta,
                fun_outer, fun_mid3d, fun_inner,
                out_outer, out_mid3d, out_inner,
                scal_arg1,
                vec_arg2_outer, vec_arg2_mid3d, vec_arg2_inner,
                scal_arg3
            )
        out_meta[META_HALO_VALID] = False

    return apply_vector


def _make_fill_halos_vector(*, jit_flags, halo, n_dims, chunker, spanner):
    set = indexers[n_dims].set

    halos = (
        (halo - 1, halo, halo),
        (halo, halo - 1, halo),
        (halo, halo, halo - 1)
    )

    @numba.njit(**jit_flags)
    def boundary_cond_vector(thread_id, meta, comp_outer, comp_mid3d, comp_inner, fun_outer, fun_mid3d, fun_inner):
        if meta[META_HALO_VALID]:
            return

        span = spanner(meta)
        rng_outer = chunker(meta, thread_id)
        last_thread = rng_outer[RNG_STOP] == span[OUTER]

        if n_dims > 2:
            pass  # TODO #96
        if n_dims > 1:
            if thread_id == 0:
                for i in range(halos[OUTER][OUTER] - 1, -1, -1):  # note: non-reverse order assumed in Extrapolated
                    for j in (INVALID_INDEX,):  # TODO #96
                        for k in range(0, span[INNER] + 2 * halos[OUTER][INNER]):
                            focus = (i, j, k)
                            set(comp_outer, i, j, k, fun_outer((focus, comp_outer), span[OUTER] + 1, SIGN_LEFT))
            if last_thread:
                for i in range(span[OUTER] + ONE_FOR_STAGGERED_GRID + halos[OUTER][OUTER],
                               span[OUTER] + ONE_FOR_STAGGERED_GRID + 2 * halos[OUTER][OUTER]):  # note: non-reverse order assumed in Extrapolated
                    for j in (INVALID_INDEX,):  # TODO #96
                        for k in range(0, span[INNER] + 2 * halos[OUTER][INNER]):
                            focus = (i, j, k)
                            set(comp_outer, i, j, k, fun_outer((focus, comp_outer), span[OUTER] + 1, SIGN_RIGHT))

        for i in range(rng_outer[RNG_START], rng_outer[RNG_STOP] + (2 * halos[INNER][OUTER] if last_thread else 0)) if n_dims > 1 else (INVALID_INDEX,):
            for j in (INVALID_INDEX,):
                for k in range(0, halos[INNER][INNER]):
                    focus = (i, j, k)
                    set(comp_inner, i, j, k, fun_inner((focus, comp_inner), span[INNER] + ONE_FOR_STAGGERED_GRID, SIGN_LEFT))
                for k in range(span[INNER] + 1 + halos[INNER][INNER], span[INNER] + ONE_FOR_STAGGERED_GRID + 2 * halos[INNER][INNER]):
                    focus = (i, j, k)
                    set(comp_inner, i, j, k, fun_inner((focus, comp_inner), span[INNER] + ONE_FOR_STAGGERED_GRID, SIGN_RIGHT))

        if n_dims > 1:
            for i in range(rng_outer[RNG_START], rng_outer[RNG_STOP] + ((ONE_FOR_STAGGERED_GRID + 2 * halos[OUTER][OUTER]) if last_thread else 0)):
                for j in (INVALID_INDEX,):
                    for k in range(0, halos[OUTER][INNER]):
                        focus = (i, j, k)
                        set(comp_outer, i, j, k, fun_inner((focus, comp_outer), span[INNER], SIGN_LEFT))
                    for k in range(span[INNER] + halos[OUTER][INNER], span[INNER] + 2 * halos[OUTER][INNER]):
                        focus = (i, j, k)
                        set(comp_outer, i, j, k, fun_inner((focus, comp_outer), span[INNER], SIGN_RIGHT))

        if n_dims > 1:
            if thread_id == 0:
                for i in range(0, halos[INNER][OUTER]):
                    for j in (INVALID_INDEX,):  # TODO #96
                        for k in range(0, span[INNER] + ONE_FOR_STAGGERED_GRID + 2 * halos[INNER][INNER]):
                            focus = (i, j, k)
                            set(comp_inner, i, j, k, fun_outer((focus, comp_inner), span[OUTER], SIGN_LEFT))
            if last_thread:
                for i in range(span[OUTER] + halos[INNER][OUTER], span[OUTER] + 2 * halos[INNER][OUTER]):
                    for j in (INVALID_INDEX,):  # TODO #96
                        for k in range(0, span[INNER] + ONE_FOR_STAGGERED_GRID + 2 * halos[INNER][INNER]):
                            focus = (i, j, k)
                            set(comp_inner, i, j, k, fun_outer((focus, comp_inner), span[OUTER], SIGN_RIGHT))

    return boundary_cond_vector

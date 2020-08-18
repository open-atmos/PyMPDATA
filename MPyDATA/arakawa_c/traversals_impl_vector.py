import numba

from .indexers import indexers
from .meta import META_HALO_VALID
from .enumerations import OUTER, INNER


def _make_apply_vector(*, jit_flags, halo, n_dims, n_threads, spanner, chunker, boundary_cond_vector,
                       boundary_cond_scalar):
    set = indexers[n_dims].set

    @numba.njit(**jit_flags)
    def apply_vector_impl(thread_id, out_meta,
                          fun_outer, fun_inner,
                          out_outer, out_inner,
                          scal_arg1,
                          vec_arg2_outer, vec_arg2_inner,
                          scal_arg3
                          ):
        # TODO: use halos, points
        span = spanner(out_meta)
        rng_outer = chunker(out_meta, thread_id)
        rng_inner = (0, span[INNER])
        last_thread = rng_outer[1] == span[OUTER]
        arg2 = (vec_arg2_outer, vec_arg2_inner)

        for i in range(rng_outer[0] + halo - 1, rng_outer[1] + halo - 1 + (1 if last_thread else 0)) if n_dims > 1 else (-1,):
            for j in range(rng_inner[0] + halo - 1, rng_inner[1] + 1 + halo - 1):
                focus = (i, j)
                if n_dims > 1:
                    set(out_outer, i, j, fun_outer((focus, scal_arg1), (focus, arg2), (focus, scal_arg3)))
                set(out_inner, i, j, fun_inner((focus, scal_arg1), (focus, arg2), (focus, scal_arg3)))

    @numba.njit(**{**jit_flags, **{'parallel': n_threads > 1}})
    def apply_vector(
            fun_outer, fun_inner,
            out_meta, out_outer, out_inner,
            scal_arg1_meta, scal_arg1, scal_arg1_bc_outer, scal_arg1_bc_inner,
            vec_arg2_meta, vec_arg2_outer, vec_arg2_inner, vec_arg2_bc_outer, vec_arg2_bc_inner,
            scal_arg3_meta, scal_arg3, scal_arg3_bc_outer, scal_arg3_bc_inner
    ):
        for thread_id in range(1) if n_threads == 1 else numba.prange(n_threads):
            boundary_cond_scalar(thread_id, scal_arg1_meta, scal_arg1, scal_arg1_bc_outer, scal_arg1_bc_inner)
            boundary_cond_vector(thread_id, vec_arg2_meta, vec_arg2_outer, vec_arg2_inner, vec_arg2_bc_outer, vec_arg2_bc_inner)
            boundary_cond_scalar(thread_id, scal_arg3_meta, scal_arg3, scal_arg3_bc_outer, scal_arg3_bc_inner)
        scal_arg1_meta[META_HALO_VALID] = True
        vec_arg2_meta[META_HALO_VALID] = True
        scal_arg3_meta[META_HALO_VALID] = True

        for thread_id in range(1) if n_threads == 1 else numba.prange(n_threads):
            apply_vector_impl(
                thread_id,
                out_meta,
                fun_outer, fun_inner,
                out_outer, out_inner,
                scal_arg1,
                vec_arg2_outer, vec_arg2_inner,
                scal_arg3
            )
        out_meta[META_HALO_VALID] = False

    return apply_vector


def _make_fill_halos_vector(*, jit_flags, halo, n_dims, chunker, spanner):
    set = indexers[n_dims].set

    @numba.njit(**jit_flags)
    def boundary_cond_vector(thread_id, meta, comp_outer, comp_inner, fun_outer, fun_inner):
        if meta[META_HALO_VALID]:
            return

        # TODO: use halos, points
        span = spanner(meta)
        outer_rng = chunker(meta, thread_id)
        last_thread = outer_rng[1] == span[OUTER]
        LEFT, RIGHT = +1, -1  # TODO: somewhere esle

        if n_dims > 1:
            if thread_id == 0:
                for i in range(halo - 2, -1, -1):  # note: non-reverse order assumed in Extrapolated
                    for j in range(0, span[INNER] + 2 * halo):
                        focus = (i, j)
                        set(comp_outer, i, j, fun_outer((focus, comp_outer), span[OUTER] + 1, LEFT))
            if last_thread:
                for i in range(span[OUTER] + 1 + halo - 1,
                               span[OUTER] + 1 + 2 * (halo - 1)):  # note: non-reverse order assumed in Extrapolated
                    for j in range(0, span[INNER] + 2 * halo):
                        focus = (i, j)
                        set(comp_outer, i, j, fun_outer((focus, comp_outer), span[OUTER] + 1, RIGHT))

        for i in range(outer_rng[0], outer_rng[1] + (2 * halo if last_thread else 0)) if n_dims > 1 else (-1,):
            for j in range(0, halo - 1):
                focus = (i, j)
                set(comp_inner, i, j, fun_inner((focus, comp_inner), span[INNER] + 1, LEFT))
            for j in range(span[INNER] + 1 + halo - 1, span[INNER] + 1 + 2 * (halo - 1)):
                focus = (i, j)
                set(comp_inner, i, j, fun_inner((focus, comp_inner), span[INNER] + 1, RIGHT))

        if n_dims > 1:
            for i in range(outer_rng[0], outer_rng[1] + ((1 + 2 * (halo - 1)) if last_thread else 0)):
                for j in range(0, halo):
                    focus = (i, j)
                    set(comp_outer, i, j, fun_inner((focus, comp_outer), span[INNER], LEFT))
                for j in range(span[INNER] + halo, span[INNER] + 2 * halo):
                    focus = (i, j)
                    set(comp_outer, i, j, fun_inner((focus, comp_outer), span[INNER], RIGHT))

        if n_dims > 1:
            if thread_id == 0:
                for i in range(0, halo):
                    for j in range(0, span[INNER] + 1 + 2 * (halo - 1)):
                        focus = (i, j)
                        set(comp_inner, i, j, fun_outer((focus, comp_inner), span[OUTER], LEFT))
            if last_thread:
                for i in range(span[OUTER] + halo, span[OUTER] + 2 * halo):
                    for j in range(0, span[INNER] + 1 + 2 * (halo - 1)):
                        focus = (i, j)
                        set(comp_inner, i, j, fun_outer((focus, comp_inner), span[OUTER], RIGHT))

    return boundary_cond_vector

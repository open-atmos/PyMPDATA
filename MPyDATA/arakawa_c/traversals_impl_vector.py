import numba

from .indexers import indexers
from .meta import META_HALO_VALID


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
        span_outer, span_inner = spanner(out_meta)
        rng_outer_0, rng_outer_1 = chunker(out_meta, thread_id)
        rng_inner_0, rng_inner_1 = 0, span_inner
        last_thread = rng_outer_1 == span_outer
        arg2 = (vec_arg2_outer, vec_arg2_inner)

        for i in range(rng_outer_0 + halo - 1, rng_outer_1 + halo - 1 + (1 if last_thread else 0)) if n_dims > 1 else (-1,):
            for j in range(rng_inner_0 + halo - 1, rng_inner_1 + 1 + halo - 1):
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
        span_outer, span_inner = spanner(meta)
        rng_outer_0, rng_outer_1 = chunker(meta, thread_id)
        last_thread = rng_outer_1 == span_outer
        LEFT, RIGHT = +1, -1  # TODO: somewhere esle

        if n_dims > 1:
            if thread_id == 0:
                for i in range(halo - 2, -1, -1):  # note: non-reverse order assumed in Extrapolated
                    for j in range(0, span_inner + 2 * halo):
                        focus = (i, j)
                        set(comp_outer, i, j, fun_outer((focus, comp_outer), span_outer + 1, LEFT))
            if last_thread:
                for i in range(span_outer + 1 + halo - 1,
                               span_outer + 1 + 2 * (halo - 1)):  # note: non-reverse order assumed in Extrapolated
                    for j in range(0, span_inner + 2 * halo):
                        focus = (i, j)
                        set(comp_outer, i, j, fun_outer((focus, comp_outer), span_outer + 1, RIGHT))

        for i in range(rng_outer_0, rng_outer_1 + (2 * halo if last_thread else 0)) if n_dims > 1 else (-1,):
            for j in range(0, halo - 1):
                focus = (i, j)
                set(comp_inner, i, j, fun_inner((focus, comp_inner), span_inner + 1, LEFT))
            for j in range(span_inner + 1 + halo - 1, span_inner + 1 + 2 * (halo - 1)):
                focus = (i, j)
                set(comp_inner, i, j, fun_inner((focus, comp_inner), span_inner + 1, RIGHT))

        if n_dims > 1:
            for i in range(rng_outer_0, rng_outer_1 + ((1 + 2 * (halo - 1)) if last_thread else 0)):
                for j in range(0, halo):
                    focus = (i, j)
                    set(comp_outer, i, j, fun_inner((focus, comp_outer), span_inner, LEFT))
                for j in range(span_inner + halo, span_inner + 2 * halo):
                    focus = (i, j)
                    set(comp_outer, i, j, fun_inner((focus, comp_outer), span_inner, RIGHT))

        if n_dims > 1:
            if thread_id == 0:
                for i in range(0, halo):
                    for j in range(0, span_inner + 1 + 2 * (halo - 1)):
                        focus = (i, j)
                        set(comp_inner, i, j, fun_outer((focus, comp_inner), span_outer, LEFT))
            if last_thread:
                for i in range(span_outer + halo, span_outer + 2 * halo):
                    for j in range(0, span_inner + 1 + 2 * (halo - 1)):
                        focus = (i, j)
                        set(comp_inner, i, j, fun_outer((focus, comp_inner), span_outer, RIGHT))

    return boundary_cond_vector

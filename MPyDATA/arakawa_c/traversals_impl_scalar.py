import numba

from .indexers import indexers
from .meta import META_HALO_VALID


def _make_apply_scalar(*, loop, jit_flags, n_dims, halo, n_threads, chunker, spanner,
                       boundary_cond_vector, boundary_cond_scalar):
    set = indexers[n_dims].set
    get = indexers[n_dims].get

    if loop:
        @numba.njit(**jit_flags)
        def apply_scalar_impl(thread_id, out_meta,
                              fun_outer, fun_inner,
                              out,
                              vec_arg1_outer, vec_arg1_inner,
                              scal_arg2, scal_arg3, scal_arg4
                              ):
            # TODO: use halos and points
            span_outer, span_inner = spanner(out_meta)
            rng_outer_0, rng_outer_1 = chunker(out_meta, thread_id)
            rng_inner_0, rng_inner_1 = 0, span_inner

            vec_arg1_tpl = (vec_arg1_outer, vec_arg1_inner)
            for i in range(rng_outer_0 + halo, rng_outer_1 + halo) if n_dims > 1 else (-1,):
                for j in range(rng_inner_0 + halo, rng_inner_1 + halo):
                    focus = (i, j)
                    if n_dims > 1:
                        set(out, i, j, fun_outer(get(out, i, j), (focus, vec_arg1_tpl),
                                             (focus, scal_arg2), (focus, scal_arg3), (focus, scal_arg4)))
                    set(out, i, j, fun_inner(get(out, i, j), (focus, vec_arg1_tpl),
                                         (focus, scal_arg2), (focus, scal_arg3), (focus, scal_arg4)))
    else:
        @numba.njit(**jit_flags)
        def apply_scalar_impl(thread_id, out_meta,
                              fun, _,
                              out,
                              vec_arg1_outer, vec_arg1_inner,
                              scal_arg2, scal_arg3, scal_arg4
                              ):
            # TODO: use halos, points
            span_outer, span_inner = spanner(out_meta)
            rng_outer_0, rng_outer_1 = chunker(out_meta, thread_id)
            rng_inner_0, rng_inner_1 = 0, span_inner

            vec_arg1_tpl = (vec_arg1_outer, vec_arg1_inner)
            for i in range(rng_outer_0 + halo, rng_outer_1 + halo) if n_dims > 1 else (-1,):
                for j in range(rng_inner_0 + halo, rng_inner_1 + halo):
                    focus = (i, j)
                    set(out, i, j, fun(get(out, i, j), (focus, vec_arg1_tpl),
                                         (focus, scal_arg2), (focus, scal_arg3), (focus, scal_arg4)))

    @numba.njit(**{**jit_flags, **{'parallel': n_threads > 1}})
    def apply_scalar(fun_outer, fun_inner,
                     out_meta, out,
                     vec_arg1_meta, vec_arg1_outer, vec_arg1_inner, vec_arg1_bc_outer, vec_arg1_bc_inner,
                     scal_arg2_meta, scal_arg_2, scal_arg2_bc_outer, scal_arg2_bc_inner,
                     scal_arg3_meta, scal_arg_3, scal_arg3_bc_outer, scal_arg3_bc_inner,
                     scal_arg4_meta, scal_arg_4, scal_arg4_bc_outer, scal_arg4_bc_inner):
        for thread_id in range(1) if n_threads == 1 else numba.prange(n_threads):
            boundary_cond_vector(thread_id, vec_arg1_meta, vec_arg1_outer, vec_arg1_inner, vec_arg1_bc_outer, vec_arg1_bc_inner)
            boundary_cond_scalar(thread_id, scal_arg2_meta, scal_arg_2, scal_arg2_bc_outer, scal_arg2_bc_inner)
            boundary_cond_scalar(thread_id, scal_arg3_meta, scal_arg_3, scal_arg3_bc_outer, scal_arg3_bc_inner)
            boundary_cond_scalar(thread_id, scal_arg4_meta, scal_arg_4, scal_arg4_bc_outer, scal_arg4_bc_inner)
        vec_arg1_meta[META_HALO_VALID] = True
        scal_arg2_meta[META_HALO_VALID] = True
        scal_arg3_meta[META_HALO_VALID] = True
        scal_arg4_meta[META_HALO_VALID] = True

        for thread_id in range(1) if n_threads == 1 else numba.prange(n_threads):
            apply_scalar_impl(
                thread_id, out_meta,
                fun_outer, fun_inner, out, vec_arg1_outer, vec_arg1_inner, scal_arg_2, scal_arg_3, scal_arg_4)
        out_meta[META_HALO_VALID] = False

    return apply_scalar


def _make_fill_halos_scalar(*, jit_flags, halo, n_dims, chunker, spanner):
    set = indexers[n_dims].set

    @numba.njit(**jit_flags)
    def boundary_cond_scalar(thread_id, meta, psi, fun_outer, fun_inner):
        if meta[META_HALO_VALID]:
            return

        # TODO: use halos, spans
        span_outer, span_inner = spanner(meta)
        rng_outer_0, rng_outer_1 = chunker(meta, thread_id)
        last_thread = rng_outer_1 == span_outer

        if n_dims > 1:
            if thread_id == 0:
                for i in range(halo - 1, 0 - 1, -1):  # note: reverse order assumes in Extrapolated!
                    for j in range(0, span_inner + 2 * halo):
                        focus = (i, j)
                        set(psi, i, j, fun_outer((focus, psi), span_outer, 1))
            if last_thread:
                for i in range(span_outer + halo, span_outer + 2 * halo):  # note: non-reverse order assumed in Extrapolated
                    for j in range(0, span_inner + 2 * halo):
                        focus = (i, j)
                        set(psi, i, j, fun_outer((focus, psi), span_outer, -1))

        for i in range(rng_outer_0, rng_outer_1 + (2 * halo if last_thread else 0)) if n_dims > 1 else (-1,):
            for j in range(0, halo):
                focus = (i, j)
                set(psi, i, j, fun_inner((focus, psi), span_inner, 1))
        for i in range(rng_outer_0, rng_outer_1 + (2 * halo if last_thread else 0)) if n_dims > 1 else (-1,):
            for j in range(span_inner + halo, span_inner + 2 * halo):
                focus = (i, j)
                set(psi, i, j, fun_inner((focus, psi), span_inner, -1))

    return boundary_cond_scalar

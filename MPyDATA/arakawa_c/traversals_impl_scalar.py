import numba

from MPyDATA.arakawa_c.indexers import indexers
from MPyDATA.arakawa_c.meta import meta_halo_valid


def _make_apply_scalar(*, loop, jit_flags, n_dims, halo, n_threads, chunk, domain,
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
            n_outer, n_inner = domain(out_meta)
            rng_outer = chunk(out_meta, thread_id)
            rng_inner = (0, n_inner)

            vec_arg1_tpl = (vec_arg1_outer, vec_arg1_inner)
            for i in range(rng_outer[0] + halo, rng_outer[1] + halo) if n_dims > 1 else (-1,):
                for j in range(rng_inner[0] + halo, rng_inner[1] + halo):
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
            n_outer, n_inner = domain(out_meta)
            rng_outer = chunk(out_meta, thread_id)
            rng_inner = (0, n_inner)

            vec_arg1_tpl = (vec_arg1_outer, vec_arg1_inner)
            for i in range(rng_outer[0] + halo, rng_outer[1] + halo) if n_dims > 1 else (-1,):
                for j in range(rng_inner[0] + halo, rng_inner[1] + halo):
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
        vec_arg1_meta[meta_halo_valid] = True
        scal_arg2_meta[meta_halo_valid] = True
        scal_arg3_meta[meta_halo_valid] = True
        scal_arg4_meta[meta_halo_valid] = True

        for thread_id in range(1) if n_threads == 1 else numba.prange(n_threads):
            apply_scalar_impl(
                thread_id, out_meta,
                fun_outer, fun_inner, out, vec_arg1_outer, vec_arg1_inner, scal_arg_2, scal_arg_3, scal_arg_4)
        out_meta[meta_halo_valid] = False

    return apply_scalar


def _make_fill_halos_scalar(*, jit_flags, halo, n_dims, chunk, domain):
    set = indexers[n_dims].set

    @numba.njit(**jit_flags)
    def boundary_cond_scalar(thread_id, meta, psi, fun_outer, fun_inner):
        if meta[meta_halo_valid]:
            return

        n_outer, n_inner = domain(meta)
        outer_rng = chunk(meta, thread_id)
        last_thread = outer_rng[1] == n_outer


        if n_dims > 1:
            if thread_id == 0:
                for i in range(halo - 1, 0 - 1, -1):  # note: reverse order assumes in Extrapolated!
                    for j in range(0, n_inner + 2 * halo):
                        focus = (i, j)
                        set(psi, i, j, fun_outer((focus, psi), n_outer, 1))
            if thread_id == last_thread:
                for i in range(n_outer + halo, n_outer + 2 * halo):  # note: non-reverse order assumed in Extrapolated
                    for j in range(0, n_inner + 2 * halo):
                        focus = (i, j)
                        set(psi, i, j, fun_outer((focus, psi), n_outer, -1))

        for i in range(outer_rng[0], outer_rng[1] + (2 * halo if last_thread else 0)) if n_dims > 1 else (-1,):
            for j in range(0, halo):
                focus = (i, j)
                set(psi, i, j, fun_inner((focus, psi), n_inner, 1))
        for i in range(outer_rng[0], outer_rng[1] + (2 * halo if last_thread else 0)) if n_dims > 1 else (-1,):
            for j in range(n_inner + halo, n_inner + 2 * halo):
                focus = (i, j)
                set(psi, i, j, fun_inner((focus, psi), n_inner, -1))

    return boundary_cond_scalar

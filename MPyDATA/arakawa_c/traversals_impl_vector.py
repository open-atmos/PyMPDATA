import numba

from MPyDATA.arakawa_c.indexers import indexers
from MPyDATA.arakawa_c.meta import meta_halo_valid


def _make_apply_vector(*, jit_flags, halo, n_dims, n_threads, domain, chunk, boundary_cond_vector,
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
        n_outer, n_inner = domain(out_meta)
        rng_outer = chunk(out_meta, thread_id)
        rng_inner = (0, n_inner)
        last_thread = rng_outer[1] == n_outer
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
        scal_arg1_meta[meta_halo_valid] = True
        vec_arg2_meta[meta_halo_valid] = True
        scal_arg3_meta[meta_halo_valid] = True

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
        out_meta[meta_halo_valid] = False

    return apply_vector


def _make_fill_halos_vector(*, jit_flags, halo, n_dims, chunk, domain):
    set = indexers[n_dims].set

    @numba.njit(**jit_flags)
    def boundary_cond_vector(thread_id, meta, comp_outer, comp_inner, fun_outer, fun_inner):
        if meta[meta_halo_valid]:
            return

        n_outer, n_inner = domain(meta)
        outer_rng = chunk(meta, thread_id)
        last_thread = outer_rng[1] == n_outer
        left, right = +1, -1

        if n_dims > 1:
            if thread_id == 0:
                for i in range(halo - 2, -1, -1):  # note: non-reverse order assumed in Extrapolated
                    for j in range(0, n_inner + 2 * halo):
                        focus = (i, j)
                        set(comp_outer, i, j, fun_outer((focus, comp_outer), n_outer + 1, left))
            if last_thread:
                for i in range(n_outer + 1 + halo - 1,
                               n_outer + 1 + 2 * (halo - 1)):  # note: non-reverse order assumed in Extrapolated
                    for j in range(0, n_inner + 2 * halo):
                        focus = (i, j)
                        set(comp_outer, i, j, fun_outer((focus, comp_outer), n_outer + 1, right))

        for i in range(outer_rng[0], outer_rng[1] + (2 * halo if last_thread else 0)) if n_dims > 1 else (-1,):
            for j in range(0, halo - 1):
                focus = (i, j)
                set(comp_inner, i, j, fun_inner((focus, comp_inner), n_inner + 1, left))
            for j in range(n_inner + 1 + halo - 1, n_inner + 1 + 2 * (halo - 1)):
                focus = (i, j)
                set(comp_inner, i, j, fun_inner((focus, comp_inner), n_inner + 1, right))

        if n_dims > 1:
            for i in range(outer_rng[0], outer_rng[1] + ((1 + 2 * (halo - 1)) if last_thread else 0)):
                for j in range(0, halo):
                    focus = (i, j)
                    set(comp_outer, i, j, fun_inner((focus, comp_outer), n_inner, left))
                for j in range(n_inner + halo, n_inner + 2 * halo):
                    focus = (i, j)
                    set(comp_outer, i, j, fun_inner((focus, comp_outer), n_inner, right))


        if n_dims > 1:
            if thread_id == 0:
                for i in range(0, halo):
                    for j in range(0, n_inner + 1 + 2 * (halo - 1)):
                        focus = (i, j)
                        set(comp_inner, i, j, fun_outer((focus, comp_inner), n_outer, left))
            if last_thread:
                for i in range(n_outer + halo, n_outer + 2 * halo):
                    for j in range(0, n_inner + 1 + 2 * (halo - 1)):
                        focus = (i, j)
                        set(comp_inner, i, j, fun_outer((focus, comp_inner), n_outer, right))

    return boundary_cond_vector

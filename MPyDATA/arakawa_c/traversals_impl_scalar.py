import numba

from MPyDATA.arakawa_c.indexers import indexers
from MPyDATA.arakawa_c.meta import meta_halo_valid


def _make_apply_scalar(*, loop, jit_flags, n_dims, halo, n_threads, irng, grid,
                       boundary_cond_vector, boundary_cond_scalar):
    set = indexers[n_dims].set
    get = indexers[n_dims].get

    if loop:
        @numba.njit(**jit_flags)
        def apply_scalar_impl(thread_id, out_meta,
                              fun_0, fun_1,
                              out,
                              vec_arg1_0, vec_arg1_1,
                              scal_arg2, scal_arg3, scal_arg4
                              ):
            ni, nj = grid(out_meta)
            rng_0 = irng(out_meta, thread_id)
            rng_1 = (0, nj)

            vec_arg1_tpl = (vec_arg1_0, vec_arg1_1)
            for i in range(rng_0[0] + halo, rng_0[1] + halo):
                for j in range(rng_1[0] + halo, rng_1[1] + halo) if n_dims > 1 else (-1,):
                    focus = (i, j)
                    set(out, i, j, fun_0(get(out, i, j), (focus, vec_arg1_tpl),
                                         (focus, scal_arg2), (focus, scal_arg3), (focus, scal_arg4)))
                    if n_dims > 1:
                        focus = (i, j)
                        set(out, i, j, fun_1(get(out, i, j), (focus, vec_arg1_tpl),
                                             (focus, scal_arg2), (focus, scal_arg3), (focus, scal_arg4)))
    else:
        @numba.njit(**jit_flags)
        def apply_scalar_impl(thread_id, out_meta,
                              fun_0, fun_1,
                              out,
                              vec_arg1_0, vec_arg1_1,
                              scal_arg2, scal_arg3, scal_arg4
                              ):
            ni, nj = grid(out_meta)
            rng_0 = irng(out_meta, thread_id)
            rng_1 = (0, nj)

            vec_arg1_tpl = (vec_arg1_0, vec_arg1_1)
            for i in range(rng_0[0] + halo, rng_0[1] + halo):
                for j in range(rng_1[0] + halo, rng_1[1] + halo) if n_dims > 1 else (-1,):
                    focus = (i, j)
                    set(out, i, j, fun_0(get(out, i, j), (focus, vec_arg1_tpl),
                                         (focus, scal_arg2), (focus, scal_arg3), (focus, scal_arg4)))

    @numba.njit(**{**jit_flags, **{'parallel': n_threads > 1}})
    def apply_scalar(fun_0, fun_1,
                     out_meta, out,
                     vec_arg1_meta, vec_arg1_0, vec_arg1_1, vec_arg1_bc0, vec_arg1_bc1,
                     scal_arg2_meta, scal_arg_2, scal_arg2_bc0, scal_arg2_bc1,
                     scal_arg3_meta, scal_arg_3, scal_arg3_bc0, scal_arg3_bc1,
                     scal_arg4_meta, scal_arg_4, scal_arg4_bc0, scal_arg4_bc1):
        for thread_id in range(1) if n_threads == 1 else numba.prange(n_threads):
            boundary_cond_vector(thread_id, vec_arg1_meta, vec_arg1_0, vec_arg1_1, vec_arg1_bc0, vec_arg1_bc1)
            boundary_cond_scalar(thread_id, scal_arg2_meta, scal_arg_2, scal_arg2_bc0, scal_arg2_bc1)
            boundary_cond_scalar(thread_id, scal_arg3_meta, scal_arg_3, scal_arg3_bc0, scal_arg3_bc1)
            boundary_cond_scalar(thread_id, scal_arg4_meta, scal_arg_4, scal_arg4_bc0, scal_arg4_bc1)
        vec_arg1_meta[meta_halo_valid] = True
        scal_arg2_meta[meta_halo_valid] = True
        scal_arg3_meta[meta_halo_valid] = True
        scal_arg4_meta[meta_halo_valid] = True

        for thread_id in range(1) if n_threads == 1 else numba.prange(n_threads):
            apply_scalar_impl(
                thread_id, out_meta,
                fun_0, fun_1, out, vec_arg1_0, vec_arg1_1, scal_arg_2, scal_arg_3, scal_arg_4)
        out_meta[meta_halo_valid] = False

    return apply_scalar


def _make_fill_halos_scalar(*, jit_flags, halo, n_dims, irng, grid):
    set = indexers[n_dims].set

    @numba.njit(**jit_flags)
    def boundary_cond_scalar(thread_id, meta, psi, fun_0, fun_1):
        if meta[meta_halo_valid]:
            return

        ni, nj = grid(meta)
        i_rng = irng(meta, thread_id)
        last_thread = i_rng[1] == ni

        if thread_id == 0:
            for j in range(0, nj + 2 * halo) if n_dims > 1 else (-1,):
                for i in range(halo - 1, 0 - 1, -1):  # note: reverse order assumes in Extrapolated!
                    focus = (i, j)
                    set(psi, i, j, fun_0((focus, psi), ni, 1))
                for i in range(ni + halo, ni + 2 * halo):  # note: non-reverse order assumed in Extrapolated
                    focus = (i, j)
                    set(psi, i, j, fun_0((focus, psi), ni, -1))
        if n_dims > 1:
            for i in range(i_rng[0], i_rng[1] + (2 * halo if last_thread else 0)):
                for j in range(0, halo):
                    focus = (i, j)
                    set(psi, i, j, fun_1((focus, psi), nj, 1))
                for j in range(nj + halo, nj + 2 * halo):
                    focus = (i, j)
                    set(psi, i, j, fun_1((focus, psi), nj, -1))

    return boundary_cond_scalar

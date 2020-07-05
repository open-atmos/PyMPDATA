import numba

from MPyDATA.arakawa_c.indexers import indexers
from MPyDATA.arakawa_c.meta import meta_halo_valid


def _make_apply_vector(*, jit_flags, halo, n_dims, n_threads, grid, irng, boundary_cond_vector,
                       boundary_cond_scalar):
    set = indexers[n_dims].set

    @numba.njit(**jit_flags)
    def apply_vector_impl(thread_id, out_meta,
                          fun_0, fun_1,
                          out_0, out_1,
                          scal_arg1,
                          vec_arg2_0, vec_arg2_1,
                          scal_arg3
                          ):
        ni, nj = grid(out_meta)
        rng_0 = irng(out_meta, thread_id)
        rng_1 = (0, nj)
        last_thread = rng_0[1] == ni
        arg2 = (vec_arg2_0, vec_arg2_1)

        for i in range(rng_0[0] + halo - 1, rng_0[1] + halo - 1 + (1 if last_thread else 0)):
            for j in range(rng_1[0] + halo - 1, rng_1[1] + 1 + halo - 1) if n_dims > 1 else (-1,):
                focus = (i, j)
                set(out_0, i, j, fun_0((focus, scal_arg1), (focus, arg2), (focus, scal_arg3)))
                if n_dims > 1:
                    set(out_1, i, j, fun_1((focus, scal_arg1), (focus, arg2), (focus, scal_arg3)))

    @numba.njit(**{**jit_flags, **{'parallel': n_threads > 1}})
    def apply_vector(
            fun0_0, fun0_1,
            out_meta, out_0, out_1,
            scal_arg1_meta, scal_arg1, scal_arg1_bc0, scal_arg1_bc1,
            vec_arg2_meta, vec_arg2_0, vec_arg2_1, vec_arg2_bc0, vec_arg2_bc1,
            scal_arg3_meta, scal_arg3, scal_arg3_bc0, scal_arg3_bc1
    ):
        for thread_id in range(1) if n_threads == 1 else numba.prange(n_threads):
            boundary_cond_scalar(thread_id, scal_arg1_meta, scal_arg1, scal_arg1_bc0, scal_arg1_bc1)
            boundary_cond_vector(thread_id, vec_arg2_meta, vec_arg2_0, vec_arg2_1, vec_arg2_bc0, vec_arg2_bc1)
            boundary_cond_scalar(thread_id, scal_arg3_meta, scal_arg3, scal_arg3_bc0, scal_arg3_bc1)
        scal_arg1_meta[meta_halo_valid] = True
        vec_arg2_meta[meta_halo_valid] = True
        scal_arg3_meta[meta_halo_valid] = True

        for thread_id in range(1) if n_threads == 1 else numba.prange(n_threads):
            apply_vector_impl(
                thread_id,
                out_meta,
                fun0_0, fun0_1,
                out_0, out_1,
                scal_arg1,
                vec_arg2_0, vec_arg2_1,
                scal_arg3
            )
        out_meta[meta_halo_valid] = False

    return apply_vector


def _make_fill_halos_vector(*, jit_flags, halo, n_dims, irng, grid):
    set = indexers[n_dims].set

    @numba.njit(**jit_flags)
    def boundary_cond_vector(thread_id, meta, comp_0, comp_1, fun_0, fun_1):
        if meta[meta_halo_valid]:
            return

        ni, nj = grid(meta)
        i_rng = irng(meta, thread_id)
        last_thread = i_rng[1] == ni

        if thread_id == 0:
            for j in range(0, nj + 2 * halo) if n_dims > 1 else (-1,):
                for i in range(halo - 2, -1, -1):  # note: non-reverse order assumed in Extrapolated
                    focus = (i, j)
                    set(comp_0, i, j, fun_0((focus, comp_0), ni + 1, 1))
                for i in range(ni + 1 + halo - 1, ni + 1 + 2 * (halo - 1)):  # note: non-reverse order assumed in Extrapolated
                    focus = (i, j)
                    set(comp_0, i, j, fun_0((focus, comp_0), ni + 1, -1))
        if n_dims > 1:
            for i in range(i_rng[0], i_rng[1] + (2 * halo if last_thread else 0)):
                for j in range(0, halo - 1):
                    focus = (i, j)
                    set(comp_1, i, j, fun_1((focus, comp_1), nj + 1, 1))
                for j in range(nj + 1 + halo - 1, nj + 1 + 2 * (halo - 1)):
                    focus = (i, j)
                    set(comp_1, i, j, fun_1((focus, comp_1), nj + 1, -1))
            for i in range(i_rng[0], i_rng[1] + ((1 + 2 * (halo - 1)) if last_thread else 0)):
                for j in range(0, halo) if n_dims > 1 else (-1,):
                    focus = (i, j)
                    set(comp_0, i, j, fun_1((focus, comp_0), nj, 1))
                for j in range(nj + halo, nj + 2 * halo) if n_dims > 1 else (-1,):
                    focus = (i, j)
                    set(comp_0, i, j, fun_1((focus, comp_0), nj, -1))
            if thread_id == 0:
                for j in range(0, nj + 1 + 2 * (halo - 1)):
                    for i in range(0, halo):
                        focus = (i, j)
                        set(comp_1, i, j, fun_0((focus, comp_1), ni, 1))
                    for i in range(ni + halo, ni + 2 * halo):
                        focus = (i, j)
                        set(comp_1, i, j, fun_0((focus, comp_1), ni, -1))

    return boundary_cond_vector
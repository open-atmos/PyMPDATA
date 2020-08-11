import numba

from MPyDATA.arakawa_c.indexers import indexers
from MPyDATA.arakawa_c.meta import meta_halo_valid


def _make_apply_vector(*, jit_flags, halo, n_dims, n_threads, domain, chunk, boundary_cond_vector,
                       boundary_cond_scalar):
    set = indexers[n_dims].set

    @numba.njit(**jit_flags)
    def apply_vector_impl(thread_id, out_meta,
                          fun_0, fun_1, fun_2,
                          out_0, out_1, out_2,
                          scal_arg1,
                          vec_arg2_0, vec_arg2_1, vec_arg2_2,
                          scal_arg3
                          ):
        ni, nj, nk = domain(out_meta)
        rng_0 = chunk(out_meta, thread_id)
        rng_1 = (0, nj)
        rng_2 = (0, nk)
        last_thread = rng_0[1] == ni
        arg2 = (vec_arg2_0, vec_arg2_1, vec_arg2_2)

        for i in range(rng_0[0] + halo - 1, rng_0[1] + halo - 1 + (1 if last_thread else 0)):
            for j in range(rng_1[0] + halo - 1, rng_1[1] + 1 + halo - 1) if n_dims > 1 else (-1,):
                for k in range(rng_2[0] + halo - 1, rng_2[1] + 1 + halo - 1) if n_dims > 2 else (-1,):
                    focus = (i, j, k)
                    set(out_0, i, j, k, fun_0((focus, scal_arg1), (focus, arg2), (focus, scal_arg3)))
                    if n_dims > 1:
                        set(out_1, i, j, k, fun_1((focus, scal_arg1), (focus, arg2), (focus, scal_arg3)))
                        if n_dims > 2:
                            pass  # TODO

    @numba.njit(**{**jit_flags, **{'parallel': n_threads > 1}})
    def apply_vector(
            fun0_0, fun0_1, fun0_2,
            out_meta, out_0, out_1, out_2,
            scal_arg1_meta, scal_arg1, scal_arg1_bc0, scal_arg1_bc1, scal_arg1_bc2,
            vec_arg2_meta, vec_arg2_0, vec_arg2_1, vec_arg2_2, vec_arg2_bc0, vec_arg2_bc1, vec_arg2_bc2,
            scal_arg3_meta, scal_arg3, scal_arg3_bc0, scal_arg3_bc1, scal_arg3_bc2
    ):
        for thread_id in range(1) if n_threads == 1 else numba.prange(n_threads):
            boundary_cond_scalar(thread_id, scal_arg1_meta, scal_arg1, scal_arg1_bc0, scal_arg1_bc1, scal_arg1_bc2)
            boundary_cond_vector(thread_id, vec_arg2_meta, vec_arg2_0, vec_arg2_1, vec_arg2_2, vec_arg2_bc0, vec_arg2_bc1, vec_arg2_bc2)
            boundary_cond_scalar(thread_id, scal_arg3_meta, scal_arg3, scal_arg3_bc0, scal_arg3_bc1, scal_arg3_bc2)
        scal_arg1_meta[meta_halo_valid] = True
        vec_arg2_meta[meta_halo_valid] = True
        scal_arg3_meta[meta_halo_valid] = True

        for thread_id in range(1) if n_threads == 1 else numba.prange(n_threads):
            apply_vector_impl(
                thread_id,
                out_meta,
                fun0_0, fun0_1, fun0_2,
                out_0, out_1, out_2,
                scal_arg1,
                vec_arg2_0, vec_arg2_1, vec_arg2_2,
                scal_arg3
            )
        out_meta[meta_halo_valid] = False

    return apply_vector


def _make_fill_halos_vector(*, jit_flags, halo, n_dims, chunk, domain):
    set = indexers[n_dims].set

    @numba.njit(**jit_flags)
    def boundary_cond_vector(thread_id, meta, comp_0, comp_1, comp_2, fun_0, fun_1, fun_2):
        if meta[meta_halo_valid]:
            return

        ni, nj, nk = domain(meta)
        i_rng = chunk(meta, thread_id)
        last_thread = i_rng[1] == ni

        if thread_id == 0:
            for k in range(0, nk + 2 * halo) if n_dims > 2 else (-1,):
                for j in range(0, nj + 2 * halo) if n_dims > 1 else (-1,):
                    for i in range(halo - 2, -1, -1):  # note: non-reverse order assumed in Extrapolated
                        focus = (i, j, k)
                        set(comp_0, i, j, k, fun_0((focus, comp_0), ni + 1, 1))
                    for i in range(ni + 1 + halo - 1, ni + 1 + 2 * (halo - 1)):  # note: non-reverse order assumed in Extrapolated
                        focus = (i, j)
                        set(comp_0, i, j, k, fun_0((focus, comp_0), ni + 1, -1))
        if n_dims > 1:
            for i in range(i_rng[0], i_rng[1] + (2 * halo if last_thread else 0)):
                for j in range(0, halo - 1):
                    for k in range(0, nk + 2 * halo) if n_dims > 2 else (-1,):
                        focus = (i, j, k)
                        set(comp_1, i, j, k, fun_1((focus, comp_1), nj + 1, 1))
                for j in range(nj + 1 + halo - 1, nj + 1 + 2 * (halo - 1)):
                    for k in range(0, nk + 2 * halo) if n_dims > 2 else (-1,):
                        focus = (i, j, k)
                        set(comp_1, i, j, k, fun_1((focus, comp_1), nj + 1, -1))
            for i in range(i_rng[0], i_rng[1] + ((1 + 2 * (halo - 1)) if last_thread else 0)):
                for j in range(0, halo) if n_dims > 1 else (-1,):
                    for k in range(0, nk + 2 * halo) if n_dims > 2 else (-1,):
                        focus = (i, j, k)
                        set(comp_0, i, j, k, fun_1((focus, comp_0), nj, 1))
                for j in range(nj + halo, nj + 2 * halo) if n_dims > 1 else (-1,):
                    for k in range(0, nk + 2 * halo) if n_dims > 2 else (-1,):
                        focus = (i, j, k)
                        set(comp_0, i, j, k, fun_1((focus, comp_0), nj, -1))
            if thread_id == 0:
                for j in range(0, nj + 1 + 2 * (halo - 1)):
                    for i in range(0, halo):
                        for k in range(0, nk + 2 * halo) if n_dims > 2 else (-1,):
                            focus = (i, j, k)
                            set(comp_1, i, j, k, fun_0((focus, comp_1), ni, 1))
                    for i in range(ni + halo, ni + 2 * halo):
                        for k in range(0, nk + 2 * halo) if n_dims > 2 else (-1,):
                            focus = (i, j, k)
                            set(comp_1, i, j, k, fun_0((focus, comp_1), ni, -1))
            if n_dims > 2:
                pass  # TODO

    return boundary_cond_vector

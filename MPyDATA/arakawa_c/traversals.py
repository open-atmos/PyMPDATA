"""
Created at 20.03.2020
"""

import numba
from .indexers import indexers
import numpy as np

meta_halo_valid = 0
meta_ni = 1
meta_nj = 2


def make_meta(halo_valid: bool, grid):
    meta = np.empty(3, dtype=int)
    meta[meta_halo_valid] = halo_valid
    meta[meta_ni] = grid[0]
    meta[meta_nj] = grid[1] if len(grid) > 1 else 0
    return meta


def make_grid(grid):
    @numba.njit()
    def grid_static(_):
        return grid

    @numba.njit()
    def grid_dynamic(meta):
        return meta[meta_ni], meta[meta_nj]

    if grid[0] > 0:
        return grid_static
    else:
        return grid_dynamic


class Traversals:
    def __init__(self, grid, halo, jit_flags):
        self.jit_flags = jit_flags
        self.grid = make_grid((grid[0], grid[1] if len(grid) > 1 else 0))
        self.n_dims = len(grid)
        self.halo = halo
        self._boundary_cond_scalar, self._boundary_cond_vector = self.make_boundary_conditions()
        self._apply_scalar = self.make_apply_scalar(loop=False)
        self._apply_scalar_loop = self.make_apply_scalar(loop=True)
        self._apply_vector = self.make_apply_vector()

    def apply_scalar(self, *, loop):
        if loop:
            return self._apply_scalar_loop
        else:
            return self._apply_scalar

    def apply_vector(self):
        return self._apply_vector

    def make_apply_scalar(self, loop):
        jit_flags = self.jit_flags
        halo = self.halo
        n_dims = self.n_dims
        grid = self.grid
        set = indexers[self.n_dims].set
        get = indexers[self.n_dims].get
        boundary_cond_vector = self._boundary_cond_vector
        boundary_cond_scalar = self._boundary_cond_scalar

        if loop:
            @numba.njit(**jit_flags)
            def apply_scalar_impl(ni, nj,
                                  fun_0, fun_1,
                                  out,
                                  vec_arg1_0, vec_arg1_1,
                                  scal_arg2, scal_arg3, scal_arg4
                                  ):
                vec_arg1_tpl = (vec_arg1_0, vec_arg1_1)
                for i in range(halo, ni + halo):
                    for j in range(halo, nj + halo) if n_dims > 1 else (-1,):
                        focus = (i, j)
                        set(out, i, j, fun_0(get(out, i, j), (focus, vec_arg1_tpl),
                                             (focus, scal_arg2), (focus, scal_arg3), (focus, scal_arg4)))
                        if n_dims > 1:
                            focus = (i, j)
                            set(out, i, j, fun_1(get(out, i, j), (focus, vec_arg1_tpl),
                                                 (focus, scal_arg2), (focus, scal_arg3), (focus, scal_arg4)))
        else:
            @numba.njit(**jit_flags)
            def apply_scalar_impl(ni, nj,
                                  fun_0, fun_1,
                                  out,
                                  vec_arg1_0, vec_arg1_1,
                                  scal_arg2, scal_arg3, scal_arg4
                                  ):
                vec_arg1_tpl = (vec_arg1_0, vec_arg1_1)
                for i in range(halo, ni + halo):
                    for j in range(halo, nj + halo) if n_dims > 1 else (-1,):
                        focus = (i, j)
                        set(out, i, j, fun_0(get(out, i, j), (focus, vec_arg1_tpl),
                                             (focus, scal_arg2), (focus, scal_arg3), (focus, scal_arg4)))

        @numba.njit(**jit_flags)
        def apply_scalar(fun_0, fun_1,
                         out_meta, out,
                         vec_arg1_meta, vec_arg1_0, vec_arg1_1, vec_arg1_bc0, vec_arg1_bc1,
                         scal_arg2_meta, scal_arg_2, scal_arg2_bc0, scal_arg2_bc1,
                         scal_arg3_meta, scal_arg_3, scal_arg3_bc0, scal_arg3_bc1,
                         scal_arg4_meta, scal_arg_4, scal_arg4_bc0, scal_arg4_bc1):
            boundary_cond_vector(vec_arg1_meta, vec_arg1_0, vec_arg1_1, vec_arg1_bc0, vec_arg1_bc1)
            boundary_cond_scalar(scal_arg2_meta, scal_arg_2, scal_arg2_bc0, scal_arg2_bc1)
            boundary_cond_scalar(scal_arg3_meta, scal_arg_3, scal_arg3_bc0, scal_arg3_bc1)
            boundary_cond_scalar(scal_arg4_meta, scal_arg_4, scal_arg4_bc0, scal_arg4_bc1)
            ni, nj = grid(out_meta)
            apply_scalar_impl(ni, nj, fun_0, fun_1, out, vec_arg1_0, vec_arg1_1, scal_arg_2, scal_arg_3, scal_arg_4)
            out_meta[meta_halo_valid] = False

        return apply_scalar

    def make_apply_vector(self):
        jit_flags = self.jit_flags
        halo = self.halo
        n_dims = self.n_dims
        grid = self.grid
        set = indexers[self.n_dims].set
        boundary_cond_vector = self._boundary_cond_vector
        boundary_cond_scalar = self._boundary_cond_scalar

        @numba.njit(**jit_flags)
        def apply_vector_impl(ni, nj,
                              fun0_0, fun0_1,
                              out_0, out_1,
                              scal_arg1,
                              vec_arg2_0, vec_arg2_1,
                              scal_arg3
                              ):
            out_tpl = (out_0, out_1)
            arg2 = (vec_arg2_0, vec_arg2_1)

            # -1, -1
            for i in range(halo - 1, ni + 1 + halo - 1):
                for j in range(halo - 1, nj + 1 + halo - 1) if n_dims > 1 else (-1,):
                    focus = (i, j)
                    set(out_tpl[0], i, j, fun0_0((focus, scal_arg1), (focus, arg2), (focus, scal_arg3)))
                    if n_dims > 1:
                        set(out_tpl[1], i, j, fun0_1((focus, scal_arg1), (focus, arg2), (focus, scal_arg3)))

        @numba.njit(**jit_flags)
        def apply_vector(
                fun0_0, fun0_1,
                out_meta, out_0, out_1,
                scal_arg1_meta, scal_arg1, scal_arg1_bc0, scal_arg1_bc1,
                vec_arg2_meta, vec_arg2_0, vec_arg2_1, vec_arg2_bc0, vec_arg2_bc1,
                scal_arg3_meta, scal_arg3, scal_arg3_bc0, scal_arg3_bc1,
        ):
            boundary_cond_scalar(scal_arg1_meta, scal_arg1, scal_arg1_bc0, scal_arg1_bc1)
            boundary_cond_vector(vec_arg2_meta, vec_arg2_0, vec_arg2_1, vec_arg2_bc0, vec_arg2_bc1)
            boundary_cond_scalar(scal_arg3_meta, scal_arg3, scal_arg3_bc0, scal_arg3_bc1)

            ni, nj = grid(out_meta)
            apply_vector_impl(
                ni, nj,
                fun0_0, fun0_1,
                out_0, out_1,
                scal_arg1,
                vec_arg2_0, vec_arg2_1,
                scal_arg3
            )
            out_meta[meta_halo_valid] = False

        return apply_vector

    def make_boundary_conditions(self):
        jit_flags = self.jit_flags
        halo = self.halo
        n_dims = self.n_dims
        set = indexers[self.n_dims].set
        grid = self.grid

        @numba.njit(**jit_flags)
        def boundary_cond_vector(meta, comp_0, comp_1, fun_0, fun_1):
            if meta[meta_halo_valid]:
                return

            ni, nj = grid(meta)

            for i in range(halo - 2, -1, -1):  # note: non-reverse order assumed in Extrapolated
                for j in range(0, nj + 2 * halo) if n_dims > 1 else (-1,):
                    focus = (i, j)
                    set(comp_0, i, j, fun_0((focus, comp_0), ni + 1, 1))
            for i in range(ni + 1 + halo - 1, ni + 1 + 2 * (halo - 1)):  # note: non-reverse order assumed in Extrapolated
                for j in range(0, nj + 2 * halo) if n_dims > 1 else (-1,):
                    focus = (i, j)
                    set(comp_0, i, j, fun_0((focus, comp_0), ni + 1, -1))
            if n_dims > 1:
                for j in range(0, halo - 1):
                    for i in range(0, ni + 2 * halo):
                        focus = (i, j)
                        set(comp_1, i, j, fun_1((focus, comp_1), nj + 1, 1))
                for j in range(nj + 1 + halo - 1, nj + 1 + 2 * (halo - 1)):
                    for i in range(0, ni + 2 * halo):
                        focus = (i, j)
                        set(comp_1, i, j, fun_1((focus, comp_1), nj + 1, -1))

            if n_dims > 1:
                for j in range(0, halo) if n_dims > 1 else (-1,):
                    for i in range(0, ni + 1 + 2 * (halo - 1)):
                        focus = (i, j)
                        set(comp_0, i, j, fun_1((focus, comp_0), nj, 1))
                for j in range(nj + halo, nj + 2 * halo) if n_dims > 1 else (-1,):
                    for i in range(0, ni + 1 + 2 * (halo - 1)):
                        focus = (i, j)
                        set(comp_0, i, j, fun_1((focus, comp_0), nj, -1))
                for i in range(0, halo):
                    for j in range(0, nj + 1 + 2 * (halo - 1)):
                        focus = (i, j)
                        set(comp_1, i, j, fun_0((focus, comp_1), ni, 1))
                for i in range(ni + halo, ni + 2 * halo):
                    for j in range(0, nj + 1 + 2 * (halo - 1)):
                        focus = (i, j)
                        set(comp_1, i, j, fun_0((focus, comp_1), ni, -1))

            meta[meta_halo_valid] = True

        @numba.njit(**jit_flags)
        def boundary_cond_scalar(meta, psi, fun_0, fun_1):
            if meta[meta_halo_valid]:
                return

            ni, nj = grid(meta)

            for i in range(halo - 1, 0 - 1, -1):  # note: reverse order assumes in Extrapolated!
                for j in range(0, nj + 2 * halo) if n_dims > 1 else (-1,):
                    focus = (i, j)
                    set(psi, i, j, fun_0((focus, psi), ni, 1))
            for i in range(ni + halo, ni + 2 * halo):  # note: non-reverse order assumed in Extrapolated
                for j in range(0, nj + 2 * halo) if n_dims > 1 else (-1,):
                    focus = (i, j)
                    set(psi, i, j, fun_0((focus, psi), ni, -1))
            if n_dims > 1:
                for j in range(0, halo):
                    for i in range(0, ni + 2 * halo):
                        focus = (i, j)
                        set(psi, i, j, fun_1((focus, psi), nj, 1))
                for j in range(nj + halo, nj + 2 * halo):
                    for i in range(0, ni + 2 * halo):
                        focus = (i, j)
                        set(psi, i, j, fun_1((focus, psi), nj, -1))

            meta[meta_halo_valid] = True

        return boundary_cond_scalar, boundary_cond_vector


@numba.njit()
def null_scalar_formula(_, __, ___):
    return 44.


@numba.njit()
def null_vector_formula(_, __, ___, ____):
    return 666.

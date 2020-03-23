"""
Created at 20.03.2020
"""

import numba
from MPyDATA.jit_flags import jit_flags
from .utils import set_2d, set_1d, get_2d, get_1d


def make_traversals(grid, n_dims, halo):
    ni = grid[0]
    nj = grid[1] if n_dims > 1 else 0

    # TODO: move to utils
    if n_dims == 1:
        set = set_1d
        get = get_1d
    elif n_dims == 2:
        set = set_2d
        get = get_2d
    else:
        raise NotImplementedError()

    @numba.njit(**jit_flags)
    def boundary_cond_vector(halo_valid, comp_0, comp_1, fun_0, fun_1):
        if halo_valid[0]:
            return
        # TODO comp_0[i, :] and comp_1[:, j] not filled
        if n_dims > 1:
            for j in range(0, halo) if n_dims > 1 else [-1]:
                for i in range(0, ni + 1 + 2 * (halo - 1)):
                    focus = (i, j)
                    set(comp_0, i, j, fun_1((focus, comp_0), nj, 1))
            for j in range(nj + halo, nj + 2 * halo) if n_dims > 1 else [-1]:
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

        halo_valid[0] = True

    @numba.njit(**jit_flags)
    def boundary_cond_scalar(halo_valid, psi, fun_0, fun_1):
        if halo_valid[0]:
            return

        for i in range(halo-1, 0-1, -1): # TODO: reverse order assumes in Extrapolated!
            for j in range(0, nj + 2 * halo) if n_dims > 1 else [-1]:
                focus = (i, j)
                set(psi, i, j, fun_0((focus, psi), ni, 1))
        for i in range(ni + halo, ni + 2 * halo): # TODO: non-reverse order assumed in Extrapolated
            for j in range(0, nj + 2 * halo) if n_dims > 1 else [-1]:
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

        halo_valid[0] = True

    @numba.njit(**jit_flags)
    def apply_vector(
            loop, fun0_0, fun0_1, fun1_0, fun1_1,
            out_flag, out_0, out_1,
            arg1_flag, arg1, arg1_bc0, arg1_bc1,
            arg2_flag, arg2_0, arg2_1, arg2_bc0, arg2_bc1,
            arg3_flag, arg3, arg3_bc0, arg3_bc1,
    ):
        boundary_cond_scalar(arg1_flag, arg1, arg1_bc0, arg1_bc1)
        boundary_cond_vector(arg2_flag, arg2_0, arg2_1, arg2_bc0, arg2_bc1)
        boundary_cond_scalar(arg3_flag, arg3, arg3_bc0, arg3_bc1)

        apply_vector_impl(
            loop, fun0_0, fun0_1, fun1_0, fun1_1,
            out_0, out_1,
            arg1,
            arg2_0, arg2_1,
            arg3
        )
        out_flag[0] = False

    @numba.njit(**jit_flags)
    def apply_vector_impl(loop, fun0_0, fun0_1, fun1_0, fun1_1,
                          out_0, out_1,
                          arg1,
                          arg2_0, arg2_1,
                          arg3
                          ):
        out_tpl = (out_0, out_1)
        arg2 = (arg2_0, arg2_1)

        # -1, -1
        if not loop:
            for i in range(halo - 1, ni + 1 + halo - 1):
                for j in range(halo - 1, nj + 1 + halo - 1) if n_dims > 1 else [-1]:
                    focus = (i, j)
                    set(out_tpl[0], i, j, fun0_0((focus, arg1), (focus, arg2), (focus, arg3)))
                    if n_dims > 1:
                        focus = (i, j)
                        set(out_tpl[1], i, j, fun0_1((focus, arg1), (focus, arg2), (focus, arg3)))
        else:
            for i in range(halo - 1, ni + 1 + halo - 1):
                for j in range(halo - 1, nj + 1 + halo - 1) if n_dims > 1 else [-1]:
                    focus = (i, j)
                    set(out_tpl[0], i, j, fun0_0((focus, arg1), (focus, arg2), (focus, arg3)))
                    if n_dims > 1:
                        set(out_tpl[0], i, j, fun1_0((focus, arg1), (focus, arg2), (focus, arg3)))
                    if n_dims > 1:
                        focus = (i, j)
                        set(out_tpl[1], i, j, fun0_1((focus, arg1), (focus, arg2), (focus, arg3)))
                        if n_dims > 1:
                            set(out_tpl[1], i, j, fun1_1((focus, arg1), (focus, arg2), (focus, arg3)))

    @numba.njit(**jit_flags)
    def apply_scalar(fun_0, fun_1,
                     out_flag, out,
                     arg1_flag, arg1_0, arg1_1, arg1_bc0, arg1_bc1,
                     arg2_flag, arg_2, arg2_bc0, arg2_bc1):
        boundary_cond_vector(arg1_flag, arg1_0, arg1_1, arg1_bc0, arg1_bc1)
        boundary_cond_scalar(arg2_flag, arg_2, arg2_bc0, arg2_bc1)
        apply_scalar_impl(fun_0, fun_1, out, arg1_0, arg1_1, arg_2)
        out_flag[0] = False

    @numba.njit(**jit_flags)
    def apply_scalar_impl(fun_0, fun_1,
                          out,
                          arg1_0, arg1_1,
                          arg2
                          ):
        arg1_tpl = (arg1_0, arg1_1)
        for i in range(halo, ni + halo):
            for j in range(halo, nj + halo) if n_dims > 1 else [-1]:
                focus = (i, j)
                set(out, i, j, fun_0(get(out, i, j), (focus, arg1_tpl), (focus, arg2)))
                if n_dims > 1:
                    focus = (i, j)
                    set(out, i, j, fun_1(get(out, i, j), (focus, arg1_tpl), (focus, arg2)))

    return apply_scalar, apply_vector
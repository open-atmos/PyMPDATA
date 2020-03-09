"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from .arakawa_c.scalar_field import ScalarField
from .arakawa_c.vector_field import VectorField
from .formulae.flux import make_flux
from .formulae.upwind import make_upwind
from .arrays import Arrays
from MPyDATA.clock import time
import numba
from .formulae.jit_flags import jit_flags
import numpy as np


class MPDATA_old:
    def __init__(self,
                 state: ScalarField,
                 GC_field: VectorField,
                 ):
        self.arrays = Arrays(state, GC_field)
        self.formulae = {
            "flux": make_flux(),
            "upwind": make_upwind()
        }

    def step(self, nt):
        # t0 = time()

        self.arrays._prev.swap_memory(self.arrays.curr)
        # print(time() - t0, "step()")
        self.arrays._flux.apply(
            args=(self.arrays._prev, self.arrays._GC_phys)
        )
        self.arrays.curr.apply(
            args=(self.arrays._flux,),
        )
        self.arrays.curr.add(self.arrays._prev)


class MPDATA:

    def __init__(self,
                 state: ScalarField,
                 GC_field: VectorField,
                 ):
        self.arrays = Arrays(state, GC_field)
        halo = 1
        ni = state.get().shape[0]
        nj = state.get().shape[1]
        self.step_impl = make_step(ni, nj, halo)

    def step(self, nt):
        psi = self.arrays._curr._impl.data
        prev = self.arrays._prev._impl.data
        flux_0 = self.arrays._flux._impl._data_0
        flux_1 = self.arrays._flux._impl._data_1
        GC_phys_0 = self.arrays._GC_phys._impl._data_0
        GC_phys_1 = self.arrays._GC_phys._impl._data_1

        for n in [0, nt]:

            t0 = time()
            self.step_impl(n, psi, flux_0, flux_1, GC_phys_0, GC_phys_1)
            t1 = time()

            print(f"{'compilation' if n == 0 else 'runtime'}: {t1 - t0} ms")
        if nt % 2 == 1:
            self.arrays.swaped = not self.arrays.swaped


def make_step(ni, nj, halo, n_dims=2):
    f_d = 0
    f_i = f_d + 1
    f_j = f_i + 1

    @numba.njit([numba.boolean(numba.float64),
                 numba.boolean(numba.int64)])
    def _is_integral(n):
        return int(n * 2.) % 2 == 0

    @numba.njit(**jit_flags)
    def at_1d():
        pass # TODO!!

    @numba.njit(**jit_flags)
    def at_2d(focus, arr, i, j):
        if focus[f_d] == 1:
            i, j = j, i
        return arr[focus[f_i] + i, focus[f_j] + j]

    @numba.njit(**jit_flags)
    def atv_2d(focus, arrs, i, j):
        if focus[f_d] == 1:
            i, j = j, i
        if _is_integral(i):
            d = 1
            ii = int(i)
            jj = int(j - .5)
        else:
            d = 0
            ii = int(i - .5)
            jj = int(j)
        return arrs[d][focus[f_i] + ii, focus[f_j] + jj]

    if n_dims == 1:
        at = at_1d
    elif n_dims == 2:
        at = at_2d
        atv = atv_2d
    else:
        assert False

    @numba.njit(**jit_flags)
    def apply_vector(fun, rng_i, rng_j, out_0, out_1, prev, GC_phys_0, GC_phys_1):
        GC_phys_tpl = (GC_phys_0, GC_phys_1)
        out_tpl = (out_0, out_1)
        # -1, -1
        for i in rng_i:
            for j in rng_j:
                focus = (0, i, j)
                out_tpl[0][i, j] = fun(focus, prev, GC_phys_tpl)
                if n_dims > 1:
                    focus = (1, i, j)
                    out_tpl[1][i, j] = fun(focus, prev, GC_phys_tpl)

    @numba.njit(**jit_flags)
    def apply_scalar(fun, rng_i, rng_j, out, flux_0, flux_1):
        flux_tpl = (flux_0, flux_1)
        for i in rng_i:
            for j in rng_j:
                focus = (0, i, j)
                out[i, j] = fun(focus, flux_tpl, init=out[i, j])
                if n_dims > 1:
                    focus = (1, i, j)
                    out[i, j] = fun(focus, flux_tpl, init=out[i, j])

    @numba.njit(**jit_flags)
    def flux(focus, prev, GC_phys_tpl):
        return \
            maximum_0(atv(focus, GC_phys_tpl, +.5, 0)) * at(focus, prev, 0, 0) + \
            minimum_0(atv(focus, GC_phys_tpl, +.5, 0)) * at(focus, prev, 1, 0)

    @numba.njit(**jit_flags)
    def minimum_0(c):
        return (c - np.abs(c)) / 2

    @numba.njit(**jit_flags)
    def maximum_0(c):
        return (c + np.abs(c)) / 2

    @numba.njit(**jit_flags)
    def upwind(focus, flux_tpl, init):
        return init \
             + atv(focus, flux_tpl, -.5, 0) \
             - atv(focus, flux_tpl, .5, 0)

    @numba.njit(**jit_flags)
    def boundary_cond(prev):
        prev[0, :] = prev[-2, :]
        prev[:, 0] = prev[:, -2]
        prev[-1, :] = prev[1, :]
        prev[:, -1] = prev[:, 1]

    @numba.njit(**jit_flags)
    def step(nt, psi, flux_0, flux_1, GC_phys_0, GC_phys_1):
        for _ in range(nt):
            boundary_cond(psi)
            apply_vector(flux, range(ni+1), range(nj+1),
                  flux_0, flux_1, psi, GC_phys_0, GC_phys_1)
            apply_scalar(upwind, range(1, ni+1), range(1, nj+1),
                  psi, flux_0, flux_1)
    return step


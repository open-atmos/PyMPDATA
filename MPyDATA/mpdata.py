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
        curr = self.arrays._curr._impl.data
        prev = self.arrays._prev._impl.data
        flux_0 = self.arrays._flux._impl._data_0
        flux_1 = self.arrays._flux._impl._data_1
        GC_phys_0 = self.arrays._GC_phys._impl._data_0
        GC_phys_1 = self.arrays._GC_phys._impl._data_1
        print(np.amax(prev), np.amax(GC_phys_0), np.amax(GC_phys_1))

        for n in [0, nt]:

            t0 = time()
            self.step_impl(n, curr, prev, flux_0, flux_1, GC_phys_0, GC_phys_1)
            t1 = time()

            print(f"{'compilation' if n == 0 else 'runtime'}: {t1 - t0} ms")
        if nt % 2 == 1:
            self.arrays.swaped = not self.arrays.swaped


def make_step(ni, nj, halo, n_dims=2):
    @numba.njit([numba.boolean(numba.float64),
                 numba.boolean(numba.int64)])
    def _is_integral(n):
        return int(n * 2.) % 2 == 0

    @numba.njit([numba.boolean(numba.float64),
                 numba.boolean(numba.int64)])
    def _is_fractional(n):
        return int(n * 2.) % 2 == 1

    @numba.njit(**jit_flags)
    def at_1d():
        pass

    @numba.njit(**jit_flags)
    def at_2d(ij, arr, i, j):
        return arr[ij[0] + halo + i, ij[1] + halo + j]

    @numba.njit(**jit_flags)
    def atv_2d(d, ij, arrs, i, j):
        if _is_integral(i) and _is_fractional(j):
            d = 1
            ii = int(i + 1)
            jj = int(j + .5)
        else:  # _is_integral(j) and _is_fractional(i):
            d = 0
            ii = int(i + .5)
            jj = int(j + 1)
        return arrs[d][ij[0] + ii, ij[1] + jj]

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
        for i in range(-1, rng_i):
            for j in range(rng_j):
                ij = (i, j)
                d = -1
                # for d in range(n_dims):
                out_tpl[0][i+1, j+1], _ = fun(d, ij, prev, GC_phys_tpl)
        for i in range(rng_i):
            for j in range(-1, rng_j):
                ij = (i, j)
                d = -1
                # for d in range(n_dims):
                _, out_tpl[1][i+1, j+1] = fun(d, ij, prev, GC_phys_tpl)

    @numba.njit(**jit_flags)
    def apply_scalar(fun, rng_i, rng_j, out, prev, flux_0, flux_1):
        flux_tpl = (flux_0, flux_1)
        for i in rng_i:
            for j in rng_j:
                ij = (i, j)
                out[i+1, j+1] = fun(ij, prev, flux_tpl)

    @numba.njit(**jit_flags)
    def flux(d, ij, prev, GC_phys_tpl):
        return \
            maximum_0(atv(d, ij, GC_phys_tpl, +.5, 0)) * at(ij, prev, 0, 0) + \
            minimum_0(atv(d, ij, GC_phys_tpl, +.5, 0)) * at(ij, prev, 1, 0) \
            , \
            maximum_0(atv(d, ij, GC_phys_tpl, 0, +.5)) * at(ij, prev, 0, 0) + \
            minimum_0(atv(d, ij, GC_phys_tpl, 0, +.5)) * at(ij, prev, 0, 1)

    @numba.njit(**jit_flags)
    def minimum_0(c):
        return (np.abs(c) - c) / 2

    @numba.njit(**jit_flags)
    def maximum_0(c):
        return (np.abs(c) + c) / 2

    @numba.njit(**jit_flags)
    def upwind(ij, prev, flux_tpl):
        return at(ij, prev, 0, 0) \
                     + atv(-1, ij, flux_tpl, -.5,  0) \
                     - atv(-1, ij, flux_tpl,  .5,  0) \
                     + atv(-1, ij, flux_tpl,  0, -.5) \
                     - atv(-1, ij, flux_tpl,  0,  .5)

    @numba.njit(**jit_flags)
    def boundary_cond(prev):
        prev[0, :] = prev[-2, :]
        prev[:, 0] = prev[:, -2]
        prev[-1, :] = prev[1, :]
        prev[:, -1] = prev[:, 1]

    @numba.njit(**jit_flags)
    def step(nt, curr, prev, flux_0, flux_1, GC_phys_0, GC_phys_1):
        for _ in range(nt):
            curr, prev = prev, curr
            boundary_cond(prev)
            apply_vector(flux, ni, nj,
                  flux_0, flux_1, prev, GC_phys_0, GC_phys_1)
            apply_scalar(upwind, range(ni), range(nj),
                  curr, prev, flux_0, flux_1)
    return step


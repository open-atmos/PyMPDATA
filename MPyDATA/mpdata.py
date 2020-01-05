"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from .arakawa_c.scalar_field import ScalarField
from .arakawa_c.vector_field import VectorField
from .formulae import fct_utils as fct
from .options import Options
from .mpdata_formulae import MPDATAFormulae
from .arrays import Arrays
from .utils import debug_flag

if debug_flag.VALUE:
    import MPyDATA.utils.fake_numba as numba
else:
    import numba


class MPDATA:
    def __init__(self,
        opts: Options,
        state: ScalarField,
        g_factor: ScalarField,
        GC_field: VectorField,
        formulae: MPDATAFormulae
    ):
        self.arrays = Arrays(state, g_factor, GC_field, opts)
        self.formulae = formulae
        self.opts = opts

    def clone(self):
        return MPDATA(
            self.opts,
            self.arrays.curr.clone(),
            self.arrays.G.clone(),
            self.arrays.GC_phys.clone(),
            formulae=self.formulae
        )

    @numba.jit()
    def step(self, n_iters):
        self.fct_init(psi=self.arrays.curr, n_iters=n_iters)
        if self.opts.mu != 0:
            assert self.arrays.curr.dimension == 1  # TODO
            assert self.opts.nug is False

            self.arrays.GC_curr.nd_sum(self.formulae.laplacian, args=(self.arrays.curr,))
            self.arrays.GC_curr.add(self.arrays.GC_phys)
        else:
            self.arrays.GC_curr.swap_memory(self.arrays.GC_phys)

        for i in range(n_iters):
            self.arrays.prev.swap_memory(self.arrays.curr)
            self.arrays.GC_prev.swap_memory(self.arrays.GC_curr)

            if i > 0:
                self.arrays.GC_curr.nd_sum(self.formulae.antidiff, args=(self.arrays.prev, self.arrays.GC_prev))
                self.fct_adjust_antidiff(self.arrays.GC_curr, i, flux=self.arrays.GC_prev, n_iters=n_iters)
            else:
                self.arrays.GC_curr.swap_memory(self.arrays.GC_prev)

            self.upwind(i, flux=self.arrays.GC_prev)

            if i == 0 and self.opts.mu == 0:
                self.arrays.GC_phys.swap_memory(self.arrays.GC_curr)

    @numba.jit()
    def upwind(self, i: int, flux: VectorField):
        flux.nd_sum(self.formulae.flux[0 if i == 0 else 1], (self.arrays.prev, self.arrays.GC_curr))
        self.arrays.curr.nd_sum(self.formulae.upwind, (flux, self.arrays.G))
        self.arrays.curr.add(self.arrays.prev)

    @numba.jit()
    def fct_init(self, psi: ScalarField, n_iters: int):
        if n_iters == 1 or not self.opts.fct:
            return
        self.arrays.psi_min.nd_sum(fct.psi_min, args=(psi,), ext=1)
        self.arrays.psi_max.nd_sum(fct.psi_max, args=(psi,), ext=1)

    @numba.jit()
    def fct_adjust_antidiff(self, GC: VectorField, it: int, flux: VectorField, n_iters: int):
        if n_iters == 1 or not self.opts.fct:
            return
        flux.nd_sum(self.formulae.flux[0 if it == 0 else 1], (self.arrays.prev, GC), ext=1)
        self.arrays.beta_up.nd_sum(fct.beta_up, (self.arrays.prev, self.arrays.psi_max, flux, self.arrays.G), ext=1)
        self.arrays.beta_dn.nd_sum(fct.beta_dn, (self.arrays.prev, self.arrays.psi_min, flux, self.arrays.G), ext=1)
        GC.nd_sum(self.formulae.fct_GC_mono, (GC, self.arrays.beta_up, self.arrays.beta_dn))

"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from MPyDATA.arakawa_c.scalar_field import ScalarField
from MPyDATA.arakawa_c.vector_field import VectorField
from MPyDATA.formulae.antidiff import make_antidiff
from MPyDATA.formulae.flux import make_fluxes
from MPyDATA.formulae import fct_utils as fct
from MPyDATA.formulae.upwind import make_upwind
from MPyDATA.arakawa_c.operators import NdSum

from MPyDATA.options import Options

from MPyDATA_tests.utils import debug
if debug.DEBUG:
    import MPyDATA_tests.utils.fake_numba as numba
else:
    import numba


class MPDATA:
    def __init__(self, prev: ScalarField, curr: ScalarField, G: ScalarField,
                 GC_physical: VectorField, GC_antidiff: VectorField,
                 flux: VectorField,
                 psi_min: ScalarField, psi_max: ScalarField,
                 beta_up: ScalarField, beta_dn: ScalarField,
                 opts: Options, halo: int):
        self.curr: ScalarField = curr
        self.prev: ScalarField = prev
        self.G: ScalarField = G
        self.GC_physical: VectorField = GC_physical
        self.GC_antidiff: VectorField = GC_antidiff
        self.flux: VectorField = flux
        self.psi_min: ScalarField = psi_min
        self.psi_max: ScalarField = psi_max
        self.beta_up: ScalarField = beta_up
        self.beta_dn: ScalarField = beta_dn

        self.n_iters: int = opts.n_iters
        self.halo: int = halo

        self.formulae = {
            "antidiff": make_antidiff(opts),
            "flux": make_fluxes(opts),
            "upwind": make_upwind(opts)
        }

        self.opts = opts

    @numba.jit()
    def fct_init(self):
        self.psi_min += NdSum(fct.psi_min, args=(self.prev,), ext=1)
        self.psi_max += NdSum(fct.psi_max, args=(self.prev,), ext=1)

    @numba.jit()
    def fct_adjust_antidiff(self, GC: VectorField, it:int):
        self.flux += NdSum(self.formulae["flux"][it], (self.prev, GC), ext=1)
        self.beta_up += NdSum(fct.beta_up, (self.prev, self.psi_max, self.flux, self.G), ext=1)
        self.beta_dn += NdSum(fct.beta_dn, (self.prev, self.psi_min, self.flux, self.G), ext=1)
        # s.state.GCh[s.state.ih] = nm.fct_GC_mono(s.opts, s.state.GCh, s.state.psi, s.beta_up, s.beta_dn, s.state.ih)

    @numba.jit()
    def step(self):
        for i in range(self.n_iters):
            self.prev.swap_memory(self.curr)
            if i == 0:
                GC = self.GC_physical
                if self.opts.n_iters != 1 and self.opts.fct:
                    self.fct_init()
            else:
                GC = self.GC_antidiff
                GC += NdSum(self.formulae["antidiff"], args=(self.prev, self.GC_physical))
                if self.opts.n_iters != 1 and self.opts.fct:
                    self.fct_adjust_antidiff(GC, i)
            # TODO: add .zero() and make += mean what it does
            self.flux += NdSum(self.formulae["flux"][i], (self.prev, GC))
            self.curr += NdSum(self.formulae["upwind"], (self.flux, self.G))
            self.curr += self.prev


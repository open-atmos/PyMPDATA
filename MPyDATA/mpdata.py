"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from MPyDATA.fields.interfaces import IScalarField, IVectorField
from MPyDATA.fields.utils import apply
from MPyDATA.formulae.antidiff import make_antidiff
from MPyDATA.formulae.flux import make_fluxes
from MPyDATA.formulae import fct_utils as fct
from MPyDATA.formulae.upwind import make_upwind

from MPyDATA.options import Options

from MPyDATA_tests.utils import debug
if debug.DEBUG:
    import MPyDATA_tests.utils.fake_numba as numba
else:
    import numba


class MPDATA:
    def __init__(self, prev: IScalarField, curr: IScalarField, G: IScalarField,
                 GC_physical: IVectorField, GC_antidiff: IVectorField,
                 flux: IVectorField,
                 psi_min: IScalarField, psi_max: IScalarField,
                 beta_up: IScalarField, beta_dn: IScalarField,
                 opts: Options, halo: int):
        self.curr: IScalarField = curr
        self.prev: IScalarField = prev
        self.G: IVectorField = G
        self.GC_physical: IVectorField = GC_physical
        self.GC_antidiff: IVectorField = GC_antidiff
        self.flux: IVectorField = flux
        self.psi_min: IScalarField = psi_min
        self.psi_max: IScalarField = psi_max
        self.beta_up: IScalarField = beta_up
        self.beta_dn: IScalarField = beta_dn

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
        apply(fct.psi_min, output=self.psi_min, args=(self.prev,), ext=1)
        apply(fct.psi_max, output=self.psi_max, args=(self.prev,), ext=1)

    @numba.jit()
    def fct_adjust_antidiff(self, GC: IVectorField, it:int):
        apply(self.formulae["flux"][it], self.flux, (self.prev, GC), ext=1)
        apply(fct.beta_up, self.beta_up, (self.prev, self.psi_max, self.flux, self.G), ext=1)
        apply(fct.beta_dn, self.beta_dn, (self.prev, self.psi_min, self.flux, self.G), ext=1)
        # s.state.GCh[s.state.ih] = nm.fct_GC_mono(s.opts, s.state.GCh, s.state.psi, s.beta_up, s.beta_dn, s.state.ih)

    #@numba.jit()
    def step(self):
        for i in range(self.n_iters):
            self.prev.swap_memory(self.curr)
            if i == 0:
                GC = self.GC_physical
                if self.opts.n_iters != 1 and self.opts.fct:
                    self.fct_init()
            else:
                GC = self.GC_antidiff
                apply(self.formulae["antidiff"], output=GC, args=(self.prev, self.GC_physical))
                if self.opts.n_iters != 1 and self.opts.fct:
                    self.fct_adjust_antidiff(GC, i)
            apply(self.formulae["flux"][i], self.flux, (self.prev, GC))
            apply(self.formulae["upwind"], self.curr, (self.flux, self.G))
            self.curr.get()[:] += self.prev.get()[:]

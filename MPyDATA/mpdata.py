"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from .arakawa_c.scalar_field import ScalarField
from .arakawa_c.vector_field import VectorField
from .formulae.antidiff import make_antidiff
from .formulae.flux import make_fluxes
from .formulae import fct_utils as fct
from .formulae.upwind import make_upwind
from .options import Options

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

        # TODO: assert for numba decorators? (depending on value of debug.DEBUG)
        self.formulae = {
            "antidiff": make_antidiff(opts),
            "flux": make_fluxes(opts),
            "upwind": make_upwind(opts),
            "fct_GC_mono": fct.fct_GC_mono
        }

        self.opts = opts

    @numba.jit()
    def fct_init(self):
        self.psi_min.nd_sum(fct.psi_min, args=(self.prev,), ext=1)
        self.psi_max.nd_sum(fct.psi_max, args=(self.prev,), ext=1)

    @numba.jit()
    def fct_adjust_antidiff(self, GC: VectorField, it:int):
        self.flux.nd_sum(self.formulae["flux"][it], (self.prev, GC), ext=1)
        self.beta_up.nd_sum(fct.beta_up, (self.prev, self.psi_max, self.flux, self.G), ext=1)
        self.beta_dn.nd_sum(fct.beta_dn, (self.prev, self.psi_min, self.flux, self.G), ext=1)
        GC.nd_sum(self.formulae["fct_GC_mono"], (GC, self.beta_up, self.beta_dn))

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
                GC.nd_sum(self.formulae["antidiff"], args=(self.prev, self.GC_physical))
                if self.opts.n_iters != 1 and self.opts.fct:
                    self.fct_adjust_antidiff(GC, i)
            self.flux.nd_sum(self.formulae["flux"][i], (self.prev, GC))

            self.curr.nd_sum(self.formulae["upwind"], (self.flux, self.G))
            self.curr.add(self.prev)


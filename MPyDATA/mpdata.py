"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""
from MPyDATA.fields import scalar_field, vector_field

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
    def __init__(self, prev: scalar_field.Interface, curr: scalar_field.Interface, G: scalar_field.Interface,
                 GC_physical: vector_field.Interface, GC_antidiff: vector_field.Interface,
                 flux: vector_field.Interface,
                 psi_min: scalar_field.Interface, psi_max: scalar_field.Interface,
                 beta_up: scalar_field.Interface, beta_dn: scalar_field.Interface,
                 opts: Options, halo: int):
        self.curr = curr
        self.prev = prev
        self.G = G
        self.GC_physical = GC_physical
        self.GC_antidiff = GC_antidiff
        self.flux = flux
        self.psi_min = psi_min
        self.psi_max = psi_max
        self.beta_up = beta_up
        self.beta_dn = beta_dn

        self.n_iters = opts.n_iters
        self.halo = halo

        self.formulae = {
            "antidiff": make_antidiff(opts),
            "flux": make_fluxes(opts),
            "upwind": make_upwind(opts)
        }

        self.opts = opts

    def fct_init(self):
        if (self.opts.n_iters == 1) or not self.opts.fct: return
        scalar_field.apply(fct.psi_min, output=self.psi_min, args=(self.prev,), ext=1)
        scalar_field.apply(fct.psi_max, output=self.psi_max, args=(self.prev,), ext=1)

    def fct_adjust_antidiff(self, GC, it):
        if self.opts.n_iters == 1 or not self.opts.fct:
            return
        GC.fill_halos()
        vector_field.apply(self.formulae["flux"][it], self.flux, (self.prev, GC), ext=1)
        scalar_field.apply(fct.beta_up, self.beta_up, (self.prev, self.psi_max, self.flux, self.G), ext=1)
        scalar_field.apply(fct.beta_dn, self.beta_dn, (self.prev, self.psi_min, self.flux, self.G), ext=1)

        # s.state.GCh[s.state.ih] = nm.fct_GC_mono(s.opts, s.state.GCh, s.state.psi, s.beta_up, s.beta_dn, s.state.ih)

    @numba.jit()
    def step(self):
        for i in range(self.n_iters):
            self.prev.swap_memory(self.curr)
            if i == 0:
                GC = self.GC_physical
                self.fct_init()
            else:
                GC = self.GC_antidiff
                vector_field.apply(self.formulae["antidiff"], output=GC, args=(self.prev, self.GC_physical))
                self.fct_adjust_antidiff(GC, i)
            vector_field.apply(self.formulae["flux"][i], self.flux, (self.prev, GC))
            scalar_field.apply(self.formulae["upwind"], self.curr, (self.flux, self.G))
            self.curr.data += self.prev.data

    def debug_print(self):
        print()
        color = '\033[94m'
        bold = '\033[7m'
        endcolor = '\033[0m'

        shp0 = self.curr.data.shape[0]
        shp1 = self.curr.data.shape[1]

        self.GC_physical.focus(0,0)
        self.GC_physical.set_axis(0)
        self.curr.focus(0,0)
        self.curr.set_axis(0)

        print("\t"*2, end='')
        for j in range(-self.halo, shp1 - self.halo):
            print("\t{:+.1f}".format(j), end='')
            if j != shp1-self.halo-1: print("\t{:+.1f}".format(j+.5), end='')
        print()

        for i in range(-self.halo, shp0-self.halo):
            print("\t{:+.1f}".format(i), end='')
            # i,j
            for j in range(-self.halo, shp1-self.halo):
                is_scalar_halo = (
                        i < 0 or
                        j < 0 or
                        i >= shp0-2*self.halo or
                        j >= shp1-2*self.halo
                )
                is_not_vector_halo = (
                    -(self.halo-1) <= i < shp0-2*(self.halo)+(self.halo-1) and
                        -self.halo <= j < shp1-2*(self.halo)+(self.halo-1)
                )

                if is_scalar_halo:
                    print(color, end='')
                else:
                    print(bold, end='')
                svalue = '{:04.1f}'.format(self.curr.at(i,j))
                print(f"\t{svalue}", end = endcolor)

                # i+.5,j
                if (is_not_vector_halo):
                    vvalue = '{:04.1f}'.format(self.GC_physical.at(i, j + .5))
                    print(f'\t{vvalue}', end='')
                else:
                    print('\t' * 2, end='')

            print('')
            if (i < shp0-(self.halo-1)-2):
                print("\t{:+.1f}".format(i+.5), end='')
            for j in range(-self.halo, shp1 - self.halo):
                pass
                if (
                    -(self.halo-1) <= j < shp1-(self.halo-1)-2 and
                    -self.halo <= i < shp0-(self.halo-1)-2
                ):
                    vvalue = '{:04.1f}'.format(self.GC_physical.at(i + .5, j))
                    print(f"\t\t\t{vvalue}", end='')
                else:
                    print("\t" * 2, end='')
            print('')


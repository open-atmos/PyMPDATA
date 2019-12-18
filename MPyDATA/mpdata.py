"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""
from MPyDATA import opts
from MPyDATA.fields.scalar_field import ScalarField
from MPyDATA.fields.vector_field import VectorField

from MPyDATA.formulae.antidiff import make_antidiff
from MPyDATA.formulae.flux import make_flux
from MPyDATA.formulae.upwind import make_upwind

from MPyDATA.opts import Opts
import numpy as np
import numba


class MPDATA:
    def __init__(self, prev: ScalarField, curr: ScalarField, G: ScalarField,
                 GC_physical: VectorField, GC_antidiff: VectorField,
                 flux: VectorField, opts: Opts, halo: int):
        self.curr = curr
        self.prev = prev
        self.G = G
        self.GC_physical = GC_physical
        self.GC_antidiff = GC_antidiff
        self.flux = flux

        self.n_iters = opts.n_iters
        self.halo = halo


        self.formulae = {}
        self.formulae["antidiff"] = make_antidiff(opts)
        self.formulae["flux"] = []
        for it in range(self.n_iters):
            self.formulae["flux"].append(make_flux(opts, it = it))
        # self.formulae["flux"] = make_flux(opts)
        # print(len(self.formulae["flux"]))
        self.formulae["upwind"] = make_upwind(opts)


       # FCT
        if (opts.n_iters != 1) & opts.fct:
            self.psi_min = np.full_like(self.curr, np.nan)
            self.psi_max = np.full_like(self.curr, np.nan)
            self.beta_up = np.full_like(self.curr, np.nan)
            self.beta_dn = np.full_like(self.curr, np.nan)

    def fct_init(self):
        if (opts.n_iters == 1) | ~opts.fct: return



    def fct_adjust_antidiff(self):
        if opts.n_iters == 1 | ~opts.fct: return

        # bcond.vector(s.opts, s.state.GCh, s.state.ih, s.n_halo)
        #
        # ihi = s.state.ih % nm.ONE
        # s.state.flx[ihi] = nm.flux(s.opts, it, s.state.psi, s.state.GCh, ihi)
        #
        # ii = s.state.i % nm.ONE
        # s.beta_up[ii] = nm.fct_beta_up(s.state.psi, s.psi_max, s.state.flx, s.state.G, ii)
        # s.beta_dn[ii] = nm.fct_beta_dn(s.state.psi, s.psi_min, s.state.flx, s.state.G, ii)
        #
        # s.state.GCh[s.state.ih] = nm.fct_GC_mono(s.opts, s.state.GCh, s.state.psi, s.beta_up, s.beta_dn, s.state.ih)

    @numba.jit()
    def step(self):
        for i in range(self.n_iters):
            self.prev.swap_memory(self.curr)
            self.prev.fill_halos()
            if i == 0:
                GC = self.GC_physical
            else:
                self.GC_antidiff.apply(self.formulae["antidiff"], self.prev, self.GC_physical)
                GC = self.GC_antidiff
            self.flux.apply(self.formulae["flux"][i], self.prev, GC)
            self.curr.apply(self.formulae["upwind"], self.flux, self.G)
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


import numpy as np
from numpy import testing
from numerics import numerics
from state import State


class MPDATA:
    @staticmethod
    def magn(q):
        return q.to_base_units().magnitude

    def __init__(self, nr, r_min, r_max, dt, cdf_r_lambda, coord, opts):
        self.nm = numerics()
        self.h = self.nm.half
        self.opts = opts

        #   |-----o-----|-----o--...
        # i-1/2   i   i+1/2   i+1
        # x_min     x_min+dx

        n_halo = self.halo(opts)
        self.state = State(n_halo, nr,  r_min, r_max, dt, cdf_r_lambda, coord, self.nm)

        # dt
        self.dt = self.magn(dt)

        # FCT
        if opts["n_it"] != 1 and self.opts["fct"]:
            self.psi_min = np.full_like(self.state.psi, np.nan)
            self.psi_max = np.full_like(self.state.psi, np.nan)
            self.beta_up = np.full_like(self.state.psi, np.nan)
            self.beta_dn = np.full_like(self.state.psi, np.nan)

    # TODO move to numerics
    @staticmethod
    def halo(opts):
        if opts["n_it"] > 1 and (opts["dfl"] or opts["fct"] or opts["tot"]):
            n_halo = 2
        else:
            n_halo = 1
        return n_halo

    def fct_init(s):
        if s.opts["n_it"] == 1 or not s.opts["fct"]: return

        ii = s.state.i % s.nm.one
        s.psi_min[ii] = s.nm.fct_running_minimum(s.state.psi, ii)
        s.psi_max[ii] = s.nm.fct_running_maximum(s.state.psi, ii)

    def fct_adjust_antidiff(s, it):
        if s.opts["n_it"] == 1 or not s.opts["fct"]: return

        s.bccond_GC(s.state.GCh)

        if not s.opts["iga"]:
            ihi = s.state.ih % s.nm.one
            s.state.flx[ihi] = s.nm.flux(s.state.psi, s.state.GCh, ihi)
        else:
            s.state.flx[:] = s.state.GCh[:]

        ii = s.state.i % s.nm.one
        s.beta_up[ii] = s.nm.fct_beta_up(s.state.psi, s.psi_max, s.state.flx, s.state.G, ii)
        s.beta_dn[ii] = s.nm.fct_beta_dn(s.state.psi, s.psi_min, s.state.flx, s.state.G, ii)

        s.state.GCh[s.state.ih] = s.nm.fct_GC_mono(s.opts, s.state.GCh, s.state.psi, s.beta_up, s.beta_dn, s.state.ih)

    # TODO move to BC
    def bccond_GC(s, GCh):
        GCh[:s.state.ih.start] = 0
        GCh[s.state.ih.stop:] = 0

    def step(s, drdt_r_lambda):
        # MPDATA iterations
        for it in range(s.opts["n_it"]):
            # boundary cond. for psi
            s.state.psi[:s.state.i.start] = 0
            s.state.psi[s.state.i.stop:] = 0

            if it == 0:
                s.fct_init()

            # evaluate velocities
            if it == 0:
                # C = drdt * dxdr * dt / dx
                # G = 1 / dxdr
                C = s.magn(drdt_r_lambda(s.state.rh[s.state.ih])) / s.state.Gh[s.state.ih] * s.dt / s.state.dx
                s.state.GCh[s.state.ih] = s.state.Gh[s.state.ih] * C
            else:
                s.state.GCh[s.state.ih] = s.nm.GC_antidiff(s.opts, s.state.psi, s.state.GCh, s.state.G, s.state.ih)
                s.fct_adjust_antidiff(it)

            # boundary condition for GCh
            s.bccond_GC(s.state.GCh)

            # check CFL
            testing.assert_array_less(np.amax(s.state.GCh[s.state.ih] / s.state.Gh[s.state.ih]), 1)

            # computing fluxes
            if it == 0 or not s.opts["iga"]:
                s.state.flx[s.state.ih] = s.nm.flux(s.state.psi, s.state.GCh, s.state.ih)
            else:
                s.state.flx[:] = s.state.GCh[:]

            # recording sum for conservativeness check
            Gpsi_sum0 = s.state.Gpsi_sum()

            # integration
            s.state.psi[s.state.i] = s.nm.upwind(s.state.psi, s.state.flx, s.state.G, s.state.i)

            # check positive definiteness
            if s.opts["n_it"] == 1 or not s.opts["iga"]:
                assert np.amin(s.state.psi[s.state.i]) >= 0

            # check conservativeness (including outflow from the domain)
            if s.opts["n_it"] == 1 or not (s.opts["iga"] and not s.opts["fct"]):
                testing.assert_approx_equal(
                    desired=Gpsi_sum0,
                    actual=(
                            s.state.Gpsi_sum() +
                            s.state.flx[(s.state.i + s.h).stop - 1] +
                            s.state.flx[(s.state.i - s.h).start]
                    ),
                    significant=15 if not s.opts["fct"] else 5
                )

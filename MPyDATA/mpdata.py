import numpy as np
from numpy import testing

from MPyDATA import numerics as nm
from MPyDATA.state import State


class MPDATA:
    @staticmethod
    def magn(q):
        return q.to_base_units().magnitude

    def __init__(self, nr, r_min, r_max, dt, cdf_r_lambda, coord, opts):
        self.h = nm.HALF
        self.opts = opts

        #   |-----o-----|-----o--...
        # i-1/2   i   i+1/2   i+1
        # x_min     x_min+dx

        n_halo = self.halo(opts)
        self.state = State(n_halo, nr,  r_min, r_max, dt, cdf_r_lambda, coord)

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

        ii = s.state.i % nm.ONE
        s.psi_min[ii] = nm.fct_running_minimum(s.state.psi, ii)
        s.psi_max[ii] = nm.fct_running_maximum(s.state.psi, ii)

    def fct_adjust_antidiff(s, it):
        if s.opts["n_it"] == 1 or not s.opts["fct"]: return

        s.bccond_GC(s.state.GCh)

        ihi = s.state.ih % nm.ONE
        s.state.flx[ihi] = nm.flux(s.opts, it, s.state.psi, s.state.GCh, ihi)

        ii = s.state.i % nm.ONE
        s.beta_up[ii] = nm.fct_beta_up(s.state.psi, s.psi_max, s.state.flx, s.state.G, ii)
        s.beta_dn[ii] = nm.fct_beta_dn(s.state.psi, s.psi_min, s.state.flx, s.state.G, ii)

        s.state.GCh[s.state.ih] = nm.fct_GC_mono(s.opts, s.state.GCh, s.state.psi, s.beta_up, s.beta_dn, s.state.ih)

    # TODO move to BC
    def bccond_GC(s, GCh):
        GCh[:s.state.ih.start] = 0
        GCh[s.state.ih.stop:] = 0

    def step(self, drdt_r_lambda):
        # MPDATA iterations
        state = self.state
        for it in range(self.opts["n_it"]):
            # boundary cond. for psi
            state.psi[:state.i.start] = 0
            state.psi[state.i.stop:] = 0

            if it == 0:
                self.fct_init()

            # evaluate velocities
            if it == 0:
                # C = drdt * dxdr * dt / dx
                # G = 1 / dxdr
                C = self.magn(drdt_r_lambda(state.rh[state.ih])) / state.Gh[state.ih] * self.dt / state.dx
                state.GCh[state.ih] = state.Gh[state.ih] * C
            else:
                state.GCh[state.ih] = nm.GC_antidiff(self.opts, state.psi, state.GCh, state.G, state.ih)
                self.fct_adjust_antidiff(it)

            # boundary condition for GCh
            self.bccond_GC(state.GCh)

            # check CFL
            testing.assert_array_less(np.amax(state.GCh[state.ih] / state.Gh[state.ih]), 1)

            # computing fluxes
            state.flx[state.ih] = nm.flux(self.opts, it, state.psi, state.GCh, state.ih)

            # recording sum for conservativeness check
            Gpsi_sum0 = state.Gpsi_sum()

            # integration
            state.psi[state.i] = nm.upwind(state.psi, state.flx, state.G, state.i)

            self.check_positive_definiteness()
            self.check_conservativeness(Gpsi_sum0)  # (including outflow from the domain)

    def check_positive_definiteness(self):
        if self.opts["n_it"] == 1 or not self.opts["iga"]:
            assert np.amin(self.state.psi[self.state.i]) >= 0

    def check_conservativeness(self, Gpsi_sum0):
        if self.opts["n_it"] == 1 or not (self.opts["iga"] and not self.opts["fct"]):
            testing.assert_approx_equal(
                desired=Gpsi_sum0,
                actual=(
                        self.state.Gpsi_sum() +
                        self.state.flx[(self.state.i + self.h).stop - 1] +
                        self.state.flx[(self.state.i - self.h).start]
                ),
                significant=15 if not self.opts["fct"] else 5
            )

import numpy as np
from numpy import testing
from numerics import numerics


class MPDATA:
    @staticmethod
    def magn(q):
        return q.to_base_units().magnitude

    def __init__(s, nr, r_min, r_max, dt, cdf_r_lambda, coord, opts):
        s.nm = numerics()
        s.h = s.nm.hlf

        s.opts = opts

        #   |-----o-----|-----o--...
        # i-1/2   i   i+1/2   i+1
        # x_min     x_min+dx

        if opts["n_it"] > 1 and (opts["dfl"] or opts["fct"] or opts["tot"]):
            n_halo = 2
        else:
            n_halo = 1

        s.i = slice(0, nr) + n_halo * s.nm.one

        # cell-border stuff
        s.ih = s.i % s.nm.hlf

        x_unit = coord.x(r_min).to_base_units().units

        _, s.dx = np.linspace(
            s.magn(coord.x(r_min)),
            s.magn(coord.x(r_max)),
            nr + 1,
            retstep=True
        )
        s.xh = np.linspace(
            s.magn(coord.x(r_min)) - (n_halo - 1) * s.dx,
            s.magn(coord.x(r_max)) + (n_halo - 1) * s.dx,
            nr + 1 + 2 * (n_halo - 1)
        )

        s.rh = coord.r(s.xh * x_unit)
        s.Gh = 1 / s.magn(coord.dx_dr(s.rh))
        s.GCh = np.full_like(s.Gh, np.nan)

        s.flx = np.full_like(s.Gh, np.nan)

        # cell-centered stuff
        s.x = np.linspace(
            s.xh[0] - s.dx / 2,
            s.xh[-1] + s.dx / 2,
            nr + 2 * n_halo
        )
        s._r = coord.r(s.x * x_unit)

        s.G = np.full_like(s.x, np.nan)
        s.G = 1 / s.magn(coord.dx_dr(s._r))

        # dt
        s.dt = s.magn(dt)

        # psi from cdf
        s.psi = np.full_like(s.G, np.nan)
        s.psi[s.i] = (
                np.diff(s.magn(cdf_r_lambda(s.rh[s.ih])))
                /
                np.diff(s.magn(s.rh[s.ih]))
        )

        # FCT
        if opts["n_it"] != 1 and s.opts["fct"]:
            s.psi_min = np.full_like(s.psi, np.nan)
            s.psi_max = np.full_like(s.psi, np.nan)
            s.beta_up = np.full_like(s.psi, np.nan)
            s.beta_dn = np.full_like(s.psi, np.nan)

    @property
    def pdf(s):
        return s.psi[s.i]

    @property
    def r(s):
        return s._r[s.i]

    def fct_init(s):
        if s.opts["n_it"] == 1 or not s.opts["fct"]: return

        ii = s.i % s.nm.one
        s.psi_min[ii] = s.nm.fct_running_minimum(s.psi, ii)
        s.psi_max[ii] = s.nm.fct_running_maximum(s.psi, ii)

    def fct_adjust_antidiff(s, it):
        if s.opts["n_it"] == 1 or not s.opts["fct"]: return

        s.bccond_GC(s.GCh)

        if not s.opts["iga"]:
            ihi = s.ih % s.nm.one
            s.flx[ihi] = s.nm.flux(s.psi, s.GCh, ihi)
        else:
            s.flx[:] = s.GCh[:]

        ii = s.i % s.nm.one
        s.beta_up[ii] = s.nm.fct_beta_up(s.psi, s.psi_max, s.flx, s.G, ii)
        s.beta_dn[ii] = s.nm.fct_beta_dn(s.psi, s.psi_min, s.flx, s.G, ii)

        s.GCh[s.ih] = s.nm.fct_GC_mono(s.opts, s.GCh, s.psi, s.beta_up, s.beta_dn, s.ih)

    def bccond_GC(s, GCh):
        GCh[:s.ih.start] = 0
        GCh[s.ih.stop:] = 0

    def Gpsi_sum(s):
        return np.sum(s.G[s.i] * s.psi[s.i])

    def step(s, drdt_r_lambda):
        # MPDATA iterations
        for it in range(s.opts["n_it"]):
            # boundary cond. for psi
            s.psi[:s.i.start] = 0
            s.psi[s.i.stop:] = 0

            if it == 0:
                s.fct_init()

            # evaluate velocities
            if it == 0:
                # C = drdt * dxdr * dt / dx
                # G = 1 / dxdr
                C = s.magn(drdt_r_lambda(s.rh[s.ih])) / s.Gh[s.ih] * s.dt / s.dx
                s.GCh[s.ih] = s.Gh[s.ih] * C
            else:
                s.GCh[s.ih] = s.nm.GC_antidiff(s.opts, s.psi, s.GCh, s.G, s.ih)
                s.fct_adjust_antidiff(it)

            # boundary condition for GCh
            s.bccond_GC(s.GCh)

            # check CFL
            testing.assert_array_less(np.amax(s.GCh[s.ih] / s.Gh[s.ih]), 1)

            # computing fluxes
            if it == 0 or not s.opts["iga"]:
                s.flx[s.ih] = s.nm.flux(s.psi, s.GCh, s.ih)
            else:
                s.flx[:] = s.GCh[:]

            # recording sum for conservativeness check
            Gpsi_sum0 = s.Gpsi_sum()

            # integration
            s.psi[s.i] = s.nm.upwind(s.psi, s.flx, s.G, s.i)

            # check positive definiteness
            if s.opts["n_it"] == 1 or not s.opts["iga"]:
                assert np.amin(s.psi[s.i]) >= 0

            # check conservativeness (including outflow from the domain)
            if s.opts["n_it"] == 1 or not (s.opts["iga"] and not s.opts["fct"]):
                testing.assert_approx_equal(
                    desired=Gpsi_sum0,
                    actual=(
                            s.Gpsi_sum() +
                            s.flx[(s.i + s.h).stop - 1] +
                            s.flx[(s.i - s.h).start]
                    ),
                    significant=15 if not s.opts["fct"] else 5
                )

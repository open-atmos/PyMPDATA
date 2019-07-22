import numpy as np


class numerics:
    class shift():
        def __init__(self, plus, mnus):
            self.plus = plus
            self.mnus = mnus

        def __radd__(self, arg):
            return type(arg)(
                arg.start + self.plus,
                arg.stop + self.plus
            )

        def __rsub__(self, arg):
            return type(arg)(
                arg.start - self.mnus,
                arg.stop - self.mnus
            )

        def __rmod__(self, arg):
            return type(arg)(
                arg.start - self.mnus,
                arg.stop + self.plus
            )

        def __rmul__(self, arg):
            return type(self)(
                arg * self.plus,
                arg * self.mnus
            )

    def __init__(self):
        self.one = self.shift(1, 1)
        self.hlf = self.shift(0, 1)
        self.eps = 1e-8

    def flux(self, psi, GC, ih):
        o = self.one
        h = self.hlf
        i = ih + h  # TODO !!! (dziala, rozumiemy, ale brzydkie)

        return np.maximum(0, GC[i + h]) * psi[i] + np.minimum(0, GC[i + h]) * psi[i + o]

    def upwind(self, psi, flx, G, i):
        h = self.hlf
        return psi[i] - (flx[i + h] - flx[i - h]) / G[i]

    def A(self, opts, psi, i):
        o = self.one
        e = self.eps

        result = psi[i + o] - psi[i]

        if not opts["iga"]:
            result /= (psi[i + o] + psi[i] + e)
        else:
            result /= (1 + 1)

        return result

    def dfl(self, opts, GC, G, psi, i):
        h = self.hlf
        o = self.one

        result = -.5 * GC[i + h] / (G[i + o] + G[i]) * (GC[i + o + h] - GC[i - h])

        # if not opts["iga"]:# TODO!!! https://github.com/igfuw/libmpdataxx/commit/9af83ff4c2877f1a48ae44388944ff3ce344b08d
        if True:
            result *= .5 * (psi[i + o] + psi[i])

        return result

    def ndxx_psi(s, opts, psi, i):
        o = s.one

        result = 2 * (psi[i + o + o] - psi[i + o] - psi[i] + psi[i - o])

        if opts["iga"]:
            result /= (1 + 1 + 1 + 1)
        else:
            result /= (psi[i + o + o] + psi[i + o] + psi[i] + psi[i - o])

        return result

    def tot(s, opts, GC, G, psi, i):
        h = s.hlf
        o = s.one

        return s.ndxx_psi(opts, psi, i) * (
                3 * GC[i + h] * np.abs(GC[i + h]) / ((G[i + o] + G[i]) / 2)
                - 2 * GC[i + h] ** 3 / ((G[i + o] + G[i]) / 2) ** 2
                - GC[i + h]
        ) / 6

    def GC_antidiff(self, opts, psi, GC, G, ih):
        h = self.hlf
        o = self.one
        i = ih + h  # TODO !!! (dziala, rozumiemy, ale brzydkie)

        result = (np.abs(GC[i + h]) - GC[i + h] ** 2 / (.5 * (G[i + o] + G[i]))) * self.A(opts, psi, i)

        if opts["dfl"]:
            result += self.dfl(opts, GC, G, psi, i)

        if opts["tot"]:
            result += self.tot(opts, GC, G, psi, i)

        return result

    def fct_extremum(self, extremum, a1, a2, a3, a4=None):
        if a4 is None:
            return extremum(extremum(a1, a2), a3)
        return extremum(extremum(extremum(a1, a2), a3), a4)

    def fct_running_extremum(self, psi, i, extremum):
        o = self.one
        a1 = psi[i - o]
        a2 = psi[i]
        a3 = psi[i + o]

        return self.fct_extremum(extremum, a1, a2, a3)

    def fct_running_maximum(self, psi, i):
        return self.fct_running_extremum(psi, i, np.maximum)

    def fct_running_minimum(self, psi, i):
        return self.fct_running_extremum(psi, i, np.minimum)

    def fct_beta_up(self, psi, psi_max, flx, G, i):
        e = self.eps
        o = self.one
        h = self.hlf

        return (
                       (self.fct_extremum(np.maximum, psi_max[i], psi[i - o], psi[i], psi[i + o]) - psi[i]) * G[i]
               ) / (
                       np.maximum(flx[i - h], 0)
                       - np.minimum(flx[i + h], 0)
                       + e
               )

    def fct_beta_dn(self, psi, psi_min, flx, G, i):
        e = self.eps
        o = self.one
        h = self.hlf

        return (
                       (psi[i] - self.fct_extremum(np.minimum, psi_min[i], psi[i - o], psi[i], psi[i + o])) * G[i]
               ) / (
                       np.maximum(flx[i + h], 0)
                       - np.minimum(flx[i - h], 0)
                       + e
               )

    def fct_GC_mono(s, opts, GCh, psi, beta_up, beta_dn, ih):
        h = s.hlf
        o = s.one
        i = ih + h

        result = GCh[i + h] * np.where(
            # if
            GCh[i + h] > 0,
            # then
            s.fct_extremum(np.minimum,
                           1,
                           beta_dn[i],
                           beta_up[i + o]
                           ),
            # else
            s.fct_extremum(np.minimum,
                           1,
                           beta_up[i],
                           beta_dn[i + o]
                           )
        )

        return result

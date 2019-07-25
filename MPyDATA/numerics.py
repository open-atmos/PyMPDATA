import numpy as np


class Shift:
    def __init__(self, plus, minus):
        self.plus = plus
        self.minus = minus

    def __radd__(self, arg):
        return type(arg)(
            arg.start + self.plus,
            arg.stop + self.plus
        )

    def __rsub__(self, arg):
        return type(arg)(
            arg.start - self.minus,
            arg.stop - self.minus
        )

    def __rmod__(self, arg):
        return type(arg)(
            arg.start - self.minus,
            arg.stop + self.plus
        )

    def __rmul__(self, arg):
        return type(self)(
            arg * self.plus,
            arg * self.minus
        )


ONE = Shift(1, 1)
HALF = Shift(0, 1)
EPS = 1e-8

O = ONE
H = HALF


def flux(psi, GC, ih):
    i = ih + HALF  # TODO !!! (dziala, rozumiemy, ale brzydkie)
    return np.maximum(0, GC[i + HALF]) * psi[i] + np.minimum(0, GC[i + HALF]) * psi[i + ONE]


def upwind(psi, flx, G, i):
    return psi[i] - (flx[i + HALF] - flx[i - HALF]) / G[i]


def A(opts, psi, i):
    result = psi[i + ONE] - psi[i]

    if not opts["iga"]:
        result /= (psi[i + ONE] + psi[i] + EPS)
    else:
        result /= (1 + 1)

    return result


def dfl(opts, GC, G, psi, i):
    result = -.5 * GC[i + HALF] / (G[i + ONE] + G[i]) * (GC[i + ONE + HALF] - GC[i - HALF])

    # if not opts["iga"]:# TODO!!! https://github.com/igfuw/libmpdataxx/commit/9af83ff4c2877f1a48ae44388944ff3ce344b08d
    if True:  # TODO 
        result *= .5 * (psi[i + ONE] + psi[i])

    return result


def ndxx_psi(opts, psi, i):
    result = 2 * (psi[i + ONE + ONE] - psi[i + ONE] - psi[i] + psi[i - ONE])

    if opts["iga"]:
        result /= (1 + 1 + 1 + 1)
    else:
        result /= (psi[i + ONE + ONE] + psi[i + ONE] + psi[i] + psi[i - ONE])

    return result


def tot(opts, GC, G, psi, i):
    return ndxx_psi(opts, psi, i) * (
            3 * GC[i + HALF] * np.abs(GC[i + HALF]) / ((G[i + ONE] + G[i]) / 2)
            - 2 * GC[i + HALF] ** 3 / ((G[i + ONE] + G[i]) / 2) ** 2
            - GC[i + HALF]
    ) / 6


def GC_antidiff(opts, psi, GC, G, ih):
    i = ih + HALF  # TODO !!! (dziala, rozumiemy, ale brzydkie)

    result = (np.abs(GC[i + HALF]) - GC[i + HALF] ** 2 / (.5 * (G[i + ONE] + G[i]))) * A(opts, psi, i)

    if opts["dfl"]:
        result += dfl(opts, GC, G, psi, i)

    if opts["tot"]:
        result += tot(opts, GC, G, psi, i)

    return result


def fct_extremum(extremum, a1, a2, a3, a4=None):
    if a4 is None:
        return extremum(extremum(a1, a2), a3)
    return extremum(extremum(extremum(a1, a2), a3), a4)


def fct_running_extremum(psi, i, extremum):
    a1 = psi[i - ONE]
    a2 = psi[i]
    a3 = psi[i + ONE]

    return fct_extremum(extremum, a1, a2, a3)


def fct_running_maximum(psi, i):
    return fct_running_extremum(psi, i, np.maximum)


def fct_running_minimum(psi, i):
    return fct_running_extremum(psi, i, np.minimum)


def fct_beta_up(psi, psi_max, flx, G, i):
    return (
                   (fct_extremum(np.maximum, psi_max[i], psi[i - ONE], psi[i], psi[i + ONE]) - psi[i]) * G[i]
           ) / (
                   np.maximum(flx[i - HALF], 0)
                   - np.minimum(flx[i + HALF], 0)
                   + EPS
           )


def fct_beta_dn(psi, psi_min, flx, G, i):
    return (
                   (psi[i] - fct_extremum(np.minimum, psi_min[i], psi[i - ONE], psi[i], psi[i + ONE])) * G[i]
           ) / (
                   np.maximum(flx[i + HALF], 0)
                   - np.minimum(flx[i - HALF], 0)
                   + EPS
           )


def fct_GC_mono(opts, GCh, psi, beta_up, beta_dn, ih):
    i = ih + HALF

    result = GCh[i + HALF] * np.where(
        # if
        GCh[i + HALF] > 0,
        # then
        fct_extremum(np.minimum,
                       1,
                       beta_dn[i],
                       beta_up[i + ONE]
                       ),
        # else
        fct_extremum(np.minimum,
                       1,
                       beta_up[i],
                       beta_dn[i + ONE]
                       )
    )

    return result

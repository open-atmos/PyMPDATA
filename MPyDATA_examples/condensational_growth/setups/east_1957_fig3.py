from ..physics.Rogers_and_Yau_1989 import Rogers_drdt, Rogers_pdf
from ..physics.East_and_Marshall_1954 import SizeDistribution
from scipy import integrate
import numpy as np


def _mixrat(si, pdf):
    xunit = si.micrometre
    yunit = 1 / si.micrometre / si.centimetre ** 3

    def fmgn(fun, unit):
        return lambda x: fun(x * xunit).to(unit).magnitude

    r_min = .1 * si.micrometre
    while not np.isfinite(pdf(r_min).magnitude):
        r_min *= 1.01
    rho_w = 1 * si.kilogram / si.decimetre ** 3
    rho_a = 1 * si.kilogram / si.metre ** 3
    I = integrate.quad(
        fmgn(lambda r: pdf(r) * r**3, yunit * xunit ** 3),
        r_min.to(xunit).magnitude,
        np.inf
    )[0] * yunit * xunit ** 4
    return (I * 4 / 3 * np.pi * rho_w / rho_a).to(si.gram / si.kilogram)


class East1957Fig3:
    def __init__(self, si):
        self.si = si

        self.nr = 64
        self.dt = .5 * si.second
        self.r_min = 1 * si.micrometre
        self.r_max = 25 * si.micrometre

        ksi_1 = 100 * si.micrometre ** 2 / si.second
        S = 1.00075
        self.drdt = Rogers_drdt(ksi_1, S)

        self.EM = SizeDistribution(si)
        pdf0 = Rogers_pdf(self.EM.pdf, self.drdt, 0 * si.second)
        self.C = (1 * si.gram / si.kilogram) / _mixrat(si, pdf0)

    def cdf(self, r):
        return self.C * self.EM.cdf(r)

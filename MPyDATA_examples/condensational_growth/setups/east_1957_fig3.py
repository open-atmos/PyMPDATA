import pint
from scipy import integrate
import numpy as np

si = pint.UnitRegistry()
tunit = si.second
xunit = si.micrometre
yunit = 1 / si.micrometre / si.centimetre**3
cdfunit = yunit * xunit

class Rogers_drdt:
    # Rogers and Yau p. 104 (modified value)
    ksi_1 = 100 * si.micrometre ** 2 / si.second
    S = 1.00075
    ksi = (S - 1) * ksi_1

    def __call__(self, r):
        return self.ksi / r


class Rogers_pdf:
    def __init__(self, pdf, t):
        self.t = t
        self.pdf = pdf

    def __call__(self, r):
        with np.errstate(invalid='ignore'):
            arg = np.sqrt(r ** 2 - 2 * Rogers_drdt.ksi * self.t)
        return r / arg * self.pdf(arg)


def EM_cdf(r):
    return (
        175 * np.sqrt(2 * np.pi / 11) *
        np.erf(np.sqrt(22) * np.log(r / (7 * si.micrometre))/ np.log(10)) *
        np.log(10) *
        (1 / si.centimetre**3)
    )

def EM_pdf(r):
    return (
            (700 * si.micrometre) / r *
            np.exp(-22 * (np.log10(r / (7 * si.micrometre))**2)) *
            (1 / si.centimetre**3 / si.micrometre)

def fmgn(fun, unit):
    return lambda x: fun(x * xunit).to(unit).magnitude

def mixrat(pdf):
    r_min = .1 * si.micrometre
    while not np.isfinite(pdf(r_min).magnitude):
        r_min *= 1.01
    rho_w = 1 * si.kilogram / si.decimetre ** 3
    rho_a = 1 * si.kilogram / si.metre ** 3
    I = integrate.quad(
        fmgn(lambda r: pdf(r) * (r) ** 3, yunit * xunit ** 3),
        r_min.to(xunit).magnitude,
        np.inf
    )[0] * yunit * xunit ** 4
    return (I * 4 / 3 * np.pi * rho_w / rho_a).to(si.gram / si.kilogram)


C = (1 * si.gram / si.kilogram) / mixrat(Rogers_pdf(EM_pdf, 0 * si.second))

class East1957Fig3:
    nr = 64
    dt = .5 * si.second
    r_min = 1 * si.micrometre
    r_max = 25 * si.micrometre

    drdt = Rogers_drdt()
    cdf = lambda r: C * EM_cdf(r)

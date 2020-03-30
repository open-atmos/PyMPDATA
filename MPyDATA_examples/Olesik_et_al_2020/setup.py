from MPyDATA_examples.Olesik_et_al_2020.physics import equilibrium_drop_growth
from MPyDATA_examples.Olesik_et_al_2020.physics import East_and_Marshall_1954
from scipy import integrate
import numpy as np
import pint



# based on Fig. 3 from East 1957
class Setup:
    def __init__(self, nr, dt):
        si = pint.UnitRegistry()
        self.si = si
        self.nr = nr
        self.dt = dt * si.second
        self.r_min = 1 * si.micrometre
        self.r_max = 26 * si.micrometre
        self.rho_w = 1 * si.kilogram / si.decimetre ** 3
        self.rho_a = 1 * si.kilogram / si.metre ** 3
        self.mixing_ratios = np.array([1, 2, 4, 10]) * si.gram / si.kilogram
        ts = np.array([0, 4.95, 12.4, 29.2]) * 60 * si.second
        ksi_1 = 100 * si.micrometre ** 2 / si.second
        S = 1.00075
        self.drdt = equilibrium_drop_growth.DrDt(ksi_1, S)
        self.size_distribution = East_and_Marshall_1954.SizeDistribution(si)

        self.C = (1 * si.gram / si.kilogram) / self.mixing_ratio(self.size_distribution.pdf)
        self.nt = []

        for i, m in enumerate(self.mixing_ratios):
            np.testing.assert_approx_equal(
                self.mixing_ratio(equilibrium_drop_growth.PdfEvolver(self.pdf, self.drdt, ts[i])),
                m, 3)
        for t in ts:
            self.nt.append(int((t / self.dt).to(si.dimensionless).magnitude))

    def mixing_ratio(self, pdf):
        # TODO!!!
        xunit = self.si.micrometre
        yunit = 1 / self.si.micrometre / self.si.centimetre ** 3

        def fmgn(fun, unit):
            return lambda x: fun(x * xunit).to(unit).magnitude

        r_min = .1 * self.si.micrometre
        while not np.isfinite(pdf(r_min).magnitude):
            r_min *= 1.01

        I = integrate.quad(
            fmgn(lambda r: pdf(r) * r ** 3, yunit * xunit ** 3),
            r_min.to(xunit).magnitude,
            np.inf
        )[0] * yunit * xunit ** 4
        return (I * 4 / 3 * np.pi * self.rho_w / self.rho_a).to(self.si.gram / self.si.kilogram)

    def pdf(self, r):
        return self.C * self.size_distribution.pdf(r)

    def cdf(self, r):
        return self.C * self.size_distribution.cdf(r)



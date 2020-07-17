from MPyDATA_examples.Olesik_et_al_2020.physics import equilibrium_drop_growth
from MPyDATA_examples.Olesik_et_al_2020.physics import East_and_Marshall_1954
from MPyDATA.arakawa_c.discretisation import discretised_analytical_solution
from scipy import integrate
import numpy as np
import pint

default_nr = 64
default_GC_max = .5
default_mixing_ratios_g_kg = np.array([1, 2, 4, 10])


# based on Fig. 3 from East 1957
class Setup:
    def __init__(self, nr=default_nr, mixing_ratios_g_kg=default_mixing_ratios_g_kg):
        si = pint.UnitRegistry()
        self.si = si
        self.nr = nr
        self.r_min = 1 * si.micrometre
        self.r_max = 25 * si.micrometre
        self.rho_w = 1 * si.kilogram / si.decimetre ** 3
        self.rho_a = 1 * si.kilogram / si.metre ** 3
        self.mixing_ratios = mixing_ratios_g_kg * si.gram / si.kilogram
        ksi_1 = 100 * si.micrometre ** 2 / si.second
        S = 1.00075
        self.drdt = equilibrium_drop_growth.DrDt(ksi_1, S)
        self.size_distribution = East_and_Marshall_1954.SizeDistribution(si)

        self.C = (1 * si.gram / si.kilogram) / self.mixing_ratio(self.size_distribution.pdf)

    def mixing_ratio(self, pdf):
        # TODO!!!
        xunit = self.si.micrometre
        yunit = 1 / self.si.micrometre / self.si.centimetre ** 3

        # def fmgn(fun, unit):
        #     return lambda x: fun(x * xunit).to(unit).magnitude

        r_min = .1 * self.si.um
        while not np.isfinite(pdf(r_min).magnitude):
            r_min *= 1.01

        r_max = 250 * self.si.um

        def pdfarg(r_nounit):
            r = r_nounit * xunit
            result = pdf(r) * r ** 3
            return result.to(yunit * xunit ** 3).magnitude

        def ipdf(func, rmin, rmax):
            # TODO: optimize quad!!!!! (and take rmax = np.inf)
            # arr = np.array([rmin.to(xunit).magnitude, rmax.to(xunit).magnitude]) * self.si.dimensionless
            # y = discretised_analytical_solution(arr, func)

            av = (rmin+rmax)/2
            y = (func(av.to(xunit).magnitude))
            I_unitless = y * (r_max - r_min).to(xunit).magnitude
            return I_unitless * yunit * xunit ** 4

        I = ipdf(func=pdfarg, rmin=r_min, rmax=r_max)
        return (I * 4 / 3 * np.pi * self.rho_w / self.rho_a).to(self.si.gram / self.si.kilogram)

    def pdf(self, r):
        return self.C * self.size_distribution.pdf(r)

    def cdf(self, r):
        return self.C * self.size_distribution.cdf(r)

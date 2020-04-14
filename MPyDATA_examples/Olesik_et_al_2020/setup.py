from MPyDATA_examples.Olesik_et_al_2020.physics import equilibrium_drop_growth
from MPyDATA_examples.Olesik_et_al_2020.physics import East_and_Marshall_1954
from scipy import integrate, optimize
import numpy as np
import pint
from functools import lru_cache

default_nr = 64
default_dt = .25 # TODO!!!
default_mixing_ratios_g_kg = np.array([1, 2, 4, 10])

# based on Fig. 3 from East 1957
class Setup:
    def __init__(self, nr=default_nr, dt=default_dt, mixing_ratios_g_kg = default_mixing_ratios_g_kg):
        si = pint.UnitRegistry()
        self.si = si
        self.nr = nr
        self.dt = dt * si.second
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


        def find_out_steps(mrs, dt, pdf, drdt):
            @lru_cache()
            def findroot(ti):
                return (mr - self.mixing_ratio(equilibrium_drop_growth.PdfEvolver(pdf, drdt, ti * t_unit))).magnitude

            out_steps = []
            for mr in mrs:
                t_unit = si.second
                t = optimize.brentq(findroot, 0, (1 * si.hour).to(t_unit).magnitude)
                out_steps.append(int(((t * t_unit)/dt).to(si.dimensionless).magnitude))
            return out_steps


        self.out_steps = find_out_steps(mrs = self.mixing_ratios, dt = self.dt, pdf=self.pdf, drdt = self.drdt)

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



from PyMPDATA_examples.Olesik_et_al_2020.physics import equilibrium_drop_growth
from PyMPDATA_examples.Olesik_et_al_2020.physics import East_and_Marshall_1954
from scipy import integrate
import numpy as np
import pint
from scipy import optimize

default_nr = 64
default_GC_max = .5
default_mixing_ratios_g_kg = np.array([1, 2, 4, 6, 8, 10])
default_opt_set = {
    'a': {'n_iters': 1},
    'b': {'n_iters': 2},
    'c': {'n_iters': 2, 'infinite_gauge': True},
    'd': {'n_iters': 2, 'infinite_gauge': True, 'flux_corrected_transport': True},
    'e': {'n_iters': 2, 'DPDC': True, 'infinite_gauge': True, 'flux_corrected_transport': True},
    'f': {'n_iters': 3, 'third_order_terms': True},
    'g': {'n_iters': 3},
    'h': {'n_iters': 3, 'third_order_terms': True, 'infinite_gauge': True, 'flux_corrected_transport': True},
}
colors = ['red', 'blue', 'crimson', 'orange', 'olive', 'navy', 'green', 'blueviolet']
colors = {key: colors.pop(0) for key in default_opt_set.keys()}


def option_string(opts):
    str_repl = [["'n_iters': 1", "upwind"],
                ["'n_iters': 2", "MPDATA 2 iterations"],
                ["'n_iters': 3", "MPDATA 3 iterations"],
                ["'", ""],
                [": True", ""],
                ["_", " "],
                ["{", ""], ["}", ""], [",", ""], ["flux corrected transport", "non-oscillatory"]]
    for repl in str_repl:
        opts = opts.replace(repl[0], repl[1])
    return opts

# based on Fig. 3 from East 1957
class Settings:
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
        self.out_times = self.find_out_steps()

    def find_out_steps(self):
        out_steps = []
        for mr in self.mixing_ratios:
            def findroot(ti):
                return (mr - self.mixing_ratio(
                    equilibrium_drop_growth.PdfEvolver(self.pdf, self.drdt, ti * t_unit))).magnitude
            t_unit = self.si.second
            t = optimize.brentq(findroot, 0, (1 * self.si.hour).to(t_unit).magnitude)
            out_steps.append(t)
        return out_steps

    def mixing_ratio(self, pdf):
        # TODO!!!
        xunit = self.si.micrometre
        yunit = 1 / self.si.micrometre / self.si.centimetre ** 3

        # def fmgn(fun, unit):
        #     return lambda x: fun(x * xunit).to(unit).magnitude

        r_min = .1 * self.si.um
        while not np.isfinite(pdf(r_min).magnitude):
            r_min *= 1.01

        def pdfarg(r_nounit):
            r = r_nounit * xunit
            result = pdf(r) * r ** 3
            return result.to(yunit * xunit ** 3).magnitude

        I = integrate.quad(pdfarg,
                           r_min.to(xunit).magnitude,
                           np.inf
                           )[0] * yunit * xunit ** 4
        return (I * 4 / 3 * np.pi * self.rho_w / self.rho_a).to(self.si.gram / self.si.kilogram)

    def pdf(self, r):
        return self.C * self.size_distribution.pdf(r)

    def cdf(self, r):
        return self.C * self.size_distribution.cdf(r)

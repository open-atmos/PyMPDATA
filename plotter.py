from matplotlib import pyplot
import numpy as np


# TODO
def ph_mgn(x):
    return x.to_base_units().magnitude


class plotter:
    def __init__(self, r_min, r_max):
        self.cdfarg, self.dcdfarg = np.linspace(
            r_min.magnitude,
            r_max.magnitude,
            512, retstep=True
        ) * r_min.units

        pyplot.figure(figsize=(8, 6))
        # pyplot.yscale('log')

    def done(self, title):
        pyplot.title(title)
        pyplot.grid()
        pyplot.legend()
        pyplot.show()

    def pdf_cdf(self, cdf):
        x = self.cdfarg[0:-1] + (self.cdfarg[1:] - self.cdfarg[0:-1]) / 2
        y = np.diff(ph_mgn(cdf(self.cdfarg))) / ph_mgn(self.dcdfarg)
        pyplot.plot(x, y)

    def pdf_pdf(self, pdf):
        x = self.cdfarg
        y = pdf(x)
        pyplot.plot(x, ph_mgn(y), 'k:')

    def step(self, algo, t):
        o = algo.opts
        label = f"t={t} n_it={o['n_it']} dfl={o['dfl']} tot={o['tot']} iga={o['iga']} fct={o['fct']}"
        pyplot.step(algo.r, algo.pdf, linestyle='-', where='mid', label=label)

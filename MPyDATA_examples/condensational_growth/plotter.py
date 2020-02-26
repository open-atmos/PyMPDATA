from matplotlib import pyplot
import numpy as np
from .setup import Setup


class Plotter:
    def __init__(self, setup, title):
        self.setup = setup
        self.cdfarg, self.dcdfarg = np.linspace(
            setup.r_min.magnitude,
            setup.r_max.magnitude,
            512, retstep=True
        )
        self.cdfarg *= setup.r_min.units
        self.dcdfarg *= setup.r_max.units

        self.fig, self.axs = pyplot.subplots(2, 1, figsize=(8, 8))
        self.style_dict = {}
        self.style_palette = ['dotted', '--', '-', '-.']
        self.axs[0].set_title(title)

        self.setup.si.setup_matplotlib()

        self.axs[0].xaxis.set_units(self.setup.si.micrometre)
        self.axs[0].yaxis.set_units(1 / self.setup.si.micrometre / self.setup.si.centimetre ** 3)

        self.axs[1].xaxis.set_units(self.setup.si.micrometre)
        self.axs[1].yaxis.set_units(1 / self.setup.si.micrometre)

        self.axs[0].grid()
        self.axs[1].grid()

    def done(self):
        pyplot.legend()
        pyplot.show()

    def analytical_pdf(self, pdf, mnorm):
        x = self.cdfarg

        # number distribution
        y = pdf(x)
        self.axs[0].plot(x, y, color='red')

        # normalised mass distribution
        y_mass = y * x**3 * 4 / 3 * np.pi * self.setup.rho_w / self.setup.rho_a / mnorm
        self.axs[1].plot(x, y_mass, color='blue')

    def numerical_pdf(self, x, y, bin_boundaries, label, mnorm):
        lbl = label
        if label not in self.style_dict:
            self.style_dict[label] = self.style_palette[len(self.style_dict)]
        else:
            lbl = ''

        # number distribution
        self.axs[0].step(
            x,
            y,
            where='mid', label=lbl, linestyle=self.style_dict[label], color='black'
        )

        # normalised mass distribution
        r1 = bin_boundaries[:-1]
        r2 = bin_boundaries[1:]

        self.axs[1].step(
            x,
            y * (r2 + r1) * (r2 ** 2 + r1 ** 2) / 4 * 4 / 3 * np.pi * self.setup.rho_w / self.setup.rho_a / mnorm,
            where='mid', label=lbl, linestyle=self.style_dict[label], color='black'
        )

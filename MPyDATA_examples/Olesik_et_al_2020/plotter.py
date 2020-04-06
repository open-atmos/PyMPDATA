from matplotlib import pyplot
import numpy as np
from .distributions import n_n


class Plotter:
    def __init__(self, setup):
        self.setup = setup
        self.cdfarg, self.dcdfarg = np.linspace(
            setup.r_min.magnitude,
            setup.r_max.magnitude,
            512, retstep=True
        )
        self.cdfarg *= setup.r_min.units
        self.dcdfarg *= setup.r_max.units

        self.fig, self.axs = pyplot.subplots(3, 1, figsize=(8, 8))
        self.style_dict = {}
        self.style_palette = [':',':', '--', '-', '-.']

        self.setup.si.setup_matplotlib()

        self.axs[0].yaxis.set_units(1 / self.setup.si.micrometre / self.setup.si.centimetre ** 3)
        self.axs[1].yaxis.set_units(1 / self.setup.si.micrometre / self.setup.si.centimetre ** 3 * self.setup.si.micrometre**2)
        self.axs[2].yaxis.set_units(1 / self.setup.si.micrometre)

        for i in range(3):
            self.axs[i].xaxis.set_units(self.setup.si.micrometre)
            self.axs[i].grid()

        self.axs[0].set_title('$dN/dr$')
        self.axs[1].set_title('$dS/dr$') # TODO: norm
        self.axs[2].set_title('$(dM/dr)/M_0$')

    def pdf_curve(self, pdf, mnorm, color='red'):
        x = self.cdfarg

        # number distribution
        y = pdf(x)  # / coord.x(x)
        self.axs[0].plot(x, y, color=color)

        # normalised surface distribution
        y_surf = y * x**2 * 4 * np.pi  # TODO: norm
        self.axs[1].plot(x, y_surf, color=color)

        # normalised mass distribution
        y_mass = y * x**3 * 4 / 3 * np.pi * self.setup.rho_w / self.setup.rho_a / mnorm
        self.axs[2].plot(x, y_mass, color=color)

    def pdf_histogram(self, x, y, bin_boundaries, label, mnorm, color='black'):
        lbl = label
        if label not in self.style_dict:
            self.style_dict[label] = self.style_palette[len(self.style_dict)]
        else:
            lbl = ''

        r1 = bin_boundaries[:-1]
        r2 = bin_boundaries[1:]

        # number distribution
        self.axs[0].step(
            x,
            n_n.to_n_n(y, r1, r2),
            where='mid', label=lbl, linestyle=self.style_dict[label], color=color
        )

        # normalised surface distribution # TODO: norm
        self.axs[1].step(
            x,
            n_n.to_n_s(y, r1, r2),
            where='mid', label=lbl, linestyle=self.style_dict[label], color=color
        )

        # normalised mass distribution
        self.axs[2].step(
            x,
            n_n.to_n_v(y, r1, r2) * self.setup.rho_w / self.setup.rho_a / mnorm,
            where='mid', label=lbl, linestyle=self.style_dict[label], color=color
        )

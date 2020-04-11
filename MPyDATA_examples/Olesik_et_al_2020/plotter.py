from matplotlib import pyplot
import numpy as np
from .distributions import n_n


class Plotter:
    def __init__(self, setup, plots=('n', 'm')):
        self.setup = setup
        self.plots = plots
        self.cdfarg, self.dcdfarg = np.linspace(
            setup.r_min.magnitude,
            setup.r_max.magnitude,
            512, retstep=True
        )
        self.cdfarg *= setup.r_min.units
        self.dcdfarg *= setup.r_max.units

        if len(plots) == 1:
            self.figsize = (14,9)
        else:
            self.figsize = (9,9)
        self.fig, self.axs = pyplot.subplots(len(plots), 1, figsize=self.figsize)
        if len(plots) == 1:
            self.axs = (self.axs,)
        self.fig.tight_layout(pad=5.0)
        self.style_dict = {}
        self.style_palette = ['-','-', '-', '-', '-.']

        self.setup.si.setup_matplotlib()

        if 'n' in plots:
            self.axs[plots.index('n')].yaxis.set_units(1 / self.setup.si.micrometre / self.setup.si.centimetre ** 3)
        if 's' in plots:
            self.axs[plots.index('s')].yaxis.set_units(1 / self.setup.si.micrometre / self.setup.si.centimetre ** 3 * self.setup.si.micrometre**2)
        if 'm' in plots:
            self.axs[plots.index('m')].yaxis.set_units(1 / self.setup.si.micrometre)

        for i in range(len(plots)):
            self.axs[i].xaxis.set_units(self.setup.si.micrometre)
            self.axs[i].grid()

        if 'n' in plots:
            self.axs[plots.index('n')].set_title('$dN/dr$')
        if 's' in plots:
            self.axs[plots.index('s')].set_title('$dS/dr$') # TODO: norm
        if 'm' in plots:
            self.axs[plots.index('m')].set_title('$(dM/dr)/M_0$')

    def pdf_curve(self, pdf, mnorm, color='black'):
        x = self.cdfarg
        y = pdf(x)

        # number distribution
        if 'n' in self.plots:
            self.axs[self.plots.index('n')].plot(x, y, color=color, linestyle=':')

        # normalised surface distribution
        if 's' in self.plots:
            y_surf = y * x**2 * 4 * np.pi  # TODO: norm
            self.axs[self.plots.index('s')].plot(x, y_surf, color=color)

        # normalised mass distribution
        if 'm' in self.plots:
            y_mass = y * x**3 * 4 / 3 * np.pi * self.setup.rho_w / self.setup.rho_a / mnorm
            self.axs[self.plots.index('m')].plot(x, y_mass, color=color, linestyle=':')

    def pdf_histogram(self, x, y, bin_boundaries, label, mnorm, color='black', linewidth = 1):
        lbl = label
        if label not in self.style_dict:
            self.style_dict[label] = self.style_palette[len(self.style_dict)]
        else:
            lbl = ''

        r1 = bin_boundaries[:-1]
        r2 = bin_boundaries[1:]

        # number distribution
        if 'n' in self.plots:
            self.axs[self.plots.index('n')].step(
                x,
                n_n.to_n_n(y, r1, r2),
                where='mid', label=lbl, linestyle=self.style_dict[label], color=color, linewidth = linewidth
            )

        # surface distribution # TODO: norm
        if 's' in self.plots:
            self.axs[self.plots.index('s')].step(
                x,
                n_n.to_n_s(y, r1, r2),
                where='mid', label=lbl, linestyle=self.style_dict[label], color=color
            )

        # normalised mass distribution
        if 'm' in self.plots:
            self.axs[self.plots.index('m')].step(
                x,
                n_n.to_n_v(y, r1, r2) * self.setup.rho_w / self.setup.rho_a / mnorm,
                where='mid', label=lbl, linestyle=self.style_dict[label], color=color, linewidth=linewidth
            )
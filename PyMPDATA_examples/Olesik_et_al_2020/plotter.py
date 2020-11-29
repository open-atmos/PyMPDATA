from matplotlib import pyplot
import numpy as np
from .distributions import n_n
import matplotlib


class Plotter:
    def __init__(self, settings, plots=('n', 'm')):
        self.settings = settings
        self.plots = plots
        self.cdfarg, self.dcdfarg = np.linspace(
            settings.r_min.magnitude,
            settings.r_max.magnitude,
            512, retstep=True
        )
        self.cdfarg *= settings.r_min.units
        self.dcdfarg *= settings.r_max.units

        matplotlib.rcParams.update({'font.size': 16})
        if len(plots) == 1:
            self.figsize = (10,6)
        else:
            self.figsize = (10,11)
        self.fig, self.axs = pyplot.subplots(len(plots), 1, figsize=self.figsize)
        if len(plots) == 1:
            self.axs = (self.axs,)
        self.fig.tight_layout(pad=5.0)
        self.style_dict = {}
        self.style_palette = ['-','-', '-', '-', '-.']

        self.settings.si.setup_matplotlib()

        if 'n' in plots:
            self.axs[plots.index('n')].yaxis.set_units(1 / self.settings.si.micrometre / self.settings.si.centimetre ** 3)
        if 'm' in plots:
            self.axs[plots.index('m')].yaxis.set_units(1 / self.settings.si.micrometre)

        for i in range(len(plots)):
            self.axs[i].xaxis.set_units(self.settings.si.micrometre)
            self.axs[i].grid()

        if 'm' in plots:
            self.axs[plots.index('m')].set_title('$(dM/dr)/M$')
        if 'n' in plots:
            self.axs[plots.index('n')].set_title('$dN/dr$')

    def pdf_curve(self, pdf, mnorm, color='black'):
        x = self.cdfarg
        y = pdf(x)

        # normalised mass distribution
        if 'm' in self.plots:
            y_mass = y * x ** 3 * 4 / 3 * np.pi * self.settings.rho_w / self.settings.rho_a / mnorm
            self.axs[self.plots.index('m')].plot(x, y_mass, color=color, linestyle=':')

        # number distribution
        if 'n' in self.plots:
            self.axs[self.plots.index('n')].plot(x, y, color=color, linestyle=':')

    def _plot(self, index, x, y, fill, lbl, color, linewidth):
        if fill:
            self.axs[self.plots.index(index)].fill_between(
                x,
                y,
                step='mid', label=lbl, color='gainsboro', alpha = 1)
        else:
            self.axs[self.plots.index(index)].step(
                x,
                y,
                where='mid', label=lbl, color=color, linewidth=linewidth
            )
        self.xlim(index)

    def pdf_histogram(self, x, y, bin_boundaries, label, mnorm, color='black', linewidth = 1, fill=True):
        r1 = bin_boundaries[:-1]
        r2 = bin_boundaries[1:]

        lbl = label
        if label not in self.style_dict:
            self.style_dict[label] = self.style_palette[len(self.style_dict)]
        else:
            lbl = ''

        # number distribution
        if 'n' in self.plots:
            self._plot('n', x, n_n.to_n_n(y, r1, r2), fill=fill, lbl=lbl, color=color, linewidth=linewidth)

        # normalised mass distribution
        if 'm' in self.plots:
            self._plot('m', x, n_n.to_n_v(y, r1, r2) * self.settings.rho_w / self.settings.rho_a / mnorm, fill=fill, color=color, linewidth=linewidth, lbl=lbl)

    def xlim(self, plot):
        self.axs[self.plots.index(plot)].set_xlim(
            (0, self.settings.r_max.magnitude)
        )


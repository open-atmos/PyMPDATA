from matplotlib import pyplot
import numpy as np


class Plotter:
    def __init__(self, r_min, r_max, title, si):
        self.cdfarg, self.dcdfarg = np.linspace(
            r_min.magnitude,
            r_max.magnitude,
            512, retstep=True
        )
        self.xunit = r_min.units
        self.yunit = None
        self.fig, self.axs = pyplot.subplots(2, 1, figsize=(8, 8))

        self.style_dict = {}
        self.style_palette = ['dotted', '--', '-', '-.']
        self.axs[0].set_title(title)
        self.si = si

    def done(self):
        self.axs[0].xaxis.set_units(self.si.micrometre)
        self.axs[0].yaxis.set_units(1 / self.si.micrometre / self.si.centimetre ** 3)

        self.axs[1].xaxis.set_units(self.si.micrometre)
        self.axs[1].yaxis.set_units(1 / self.si.micrometre)

        self.axs[0].grid()
        self.axs[1].grid()

        pyplot.legend()
        pyplot.show()

    def pdf_pdf(self, pdf, mnorm):
        x = self.cdfarg

        # number distribution
        y = pdf(x)
        self.axs[0].plot(x * xunit, y * self.yunit, color='red')

        # normalised mass distribution
        y *= self.yunit * (x * self.xunit) ** 3 * 4 / 3 * np.pi * rho_w / rho_a / mnorm
        self.axs[1].plot(x, y.to_base_units(), color='blue')

    def step(self, xy, label, t):
        lbl = label
        if label not in self.style_dict:
            self.style_dict[label] = self.style_palette[len(self.style_dict)]
        else:
            lbl = ''

        # number distribution
        self.axs[0].step(
            xy["r"] * self.xunit,
            xy["n"] * self.yunit,
            where='mid', label=lbl, linestyle=self.style_dict[label], color='black'
        )

        # normalised mass distribution
        r1 = xy["rb"][:-1] * self.xunit
        r2 = xy["rb"][1:] * self.xunit

        self.axs[1].step(
            xy["r"] * self.xunit,
            xy["n"] * self.yunit * (r2 + r1) * (r2 ** 2 + r1 ** 2) / 4 * 4 / 3 * np.pi * rho_w / rho_a / mnorm,
            where='mid', label=lbl, linestyle=self.style_dict[label], color='black'
        )

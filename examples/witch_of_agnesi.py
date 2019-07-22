import numpy as np
import pint
si = pint.UnitRegistry()
si.setup_matplotlib()

from runner import run
from plotter import plotter


class Setup:
    nr = 64
    nt = 256
    dt = 1 * si.second
    r_min = 25 * si.micrometre
    r_max = 75 * si.micrometre
    r_mid = r_min + .45 * (r_max - r_min)

    # Witch of Agnesi
    A = 1e-6
    B = 2 * si.micrometre
    cdf0 = lambda r: Setup.A * np.arctan((r - Setup.r_mid) / Setup.B)
    pdf0 = lambda r: Setup.A / Setup.B / (((r - Setup.r_mid) / Setup.B) ** 2 + 1)

    # Rogers and Yau p. 104
    ksi_1 = 68.2 * si.micrometre ** 2 / si.second
    S = 1.05
    ksi = (S - 1) * ksi_1
    drdt = lambda r: Setup.ksi / r

    @staticmethod
    def Rogers(r):
        with np.errstate(invalid='ignore'):
            arg = np.sqrt(r ** 2 - 2 * Setup.ksi * Setup.nt * Setup.dt)
        return r / arg * Setup.pdf0(arg)


np.seterr(all='raise')
data = run(Setup, si)

# TODO move somewhere
for fn in data:
    plot = plotter(Setup.r_min, Setup.r_max)
    # TODO initial and intermidate steps
    plot.pdf_cdf(Setup.cdf0)

    plot.step(data[fn][0], 0)

    plot.pdf_pdf(Setup.Rogers)

    for algo in data[fn]:
        plot.step(algo, Setup.nt)
    plot.done(fn.__name__)

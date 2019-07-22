import numpy as np
np.seterr(all='raise')

import pint
si = pint.UnitRegistry()


si.setup_matplotlib()

from coord import x_id, x_ln, x_p2
from mpdata import MPDATA




from plotter import plotter

nr = 64
nt = 256
dt = 1 * si.second
r_min = 25 * si.micrometre
r_max = 75 * si.micrometre
r_mid = r_min + .45 * (r_max - r_min)

# Witch of Agnesi
A = 1e-6
B = 2 * si.micrometre
cdf0 = lambda r: A * np.arctan((r - r_mid) / B)
pdf0 = lambda r: A / B / (((r - r_mid) / B) ** 2 + 1)

# Rogers and Yau p. 104
ksi_1 = 68.2 * si.micrometre ** 2 / si.second
S = 1.05
ksi = (S - 1) * ksi_1
drdt = lambda r: ksi / r




def Rogers(r):
    with np.errstate(invalid='ignore'):
        arg = np.sqrt(r ** 2 - 2 * ksi * nt * dt)
    return r / arg * pdf0(arg)




for fn in (x_id, x_p2, x_ln):
    plot = plotter(r_min, r_max)

    algos = (
        MPDATA(nr, r_min, r_max, dt, cdf0, fn(si), {"n_it": 1, "dfl": 0, "iga": 0, "tot": 0, "fct": 0}),
        MPDATA(nr, r_min, r_max, dt, cdf0, fn(si), {"n_it": 2, "dfl": 0, "iga": 0, "tot": 0, "fct": 0}),
        # MPDATA(nr, r_min, r_max, dt, cdf0, fn(si), {"n_it":3, "dfl":0, "iga":0, "tot":0, "fct":0}),
        # MPDATA(nr, r_min, r_max, dt, cdf0, fn(si), {"n_it":2, "dfl":1, "iga":0, "tot":0, fct":0}),
        # MPDATA(nr, r_min, r_max, dt, cdf0, fn(si), {"n_it":2, "dfl":0, "iga":1, "tot":0, "fct":0}),
        # MPDATA(nr, r_min, r_max, dt, cdf0, fn(si), {"n_it":2, "dfl":1, "iga":1, "tot":1, "fct":0}),
        MPDATA(nr, r_min, r_max, dt, cdf0, fn(si), {"n_it": 3, "dfl": 1, "iga": 1, "tot": 1, "fct": 0}),
        MPDATA(nr, r_min, r_max, dt, cdf0, fn(si), {"n_it": 3, "dfl": 1, "iga": 1, "tot": 1, "fct": 1})
    )

    plot.pdf_cdf(cdf0)

    plot.step(algos[0], 0)

    for algo in algos:
        for t in range(nt):
            algo.step(drdt)

    plot.pdf_pdf(Rogers)
    for algo in algos:
        plot.step(algo, nt)
    plot.done(fn.__name__)

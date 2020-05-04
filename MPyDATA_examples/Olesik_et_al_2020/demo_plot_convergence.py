# based on: https://github.com/igfuw/libmpdataxx/blob/master/tests/paper_2015_GMD/2_convergence_1d/plot.py
# by Anna Jaruga (copyright University of Warsaw, 2012)
# GPLv3+ (see the COPYING file or http://www.gnu.org/licenses/)
# modified by Michael Olesik & Sylwester Arabas, 2020

import numpy as np
from numpy import zeros, linspace
from math import pi, cos, sin, sqrt
from scipy.interpolate import griddata
from matplotlib.patches import Path, PathPatch
import matplotlib.pyplot as plt


def plot(nr, cour, ln_2_err, n_levels=11, ngrid=800 * 2, fontsize=20):
    x = zeros(nr.shape[0])
    y = zeros(nr.shape[0])

    theta = cour * pi / 2.
    r = np.log(1 / nr)
    r -= min(r)
    r *= 10 / max(r)
    for i in range(theta.shape[0]):
        x[i] = r[i] * cos(theta[i])
        y[i] = r[i] * sin(theta[i])

    levels = np.linspace(
        np.floor(min(ln_2_err)),
        np.ceil(max(ln_2_err)),
        n_levels
    )
    mn = 0
    mx = int(np.ceil(max(r)))

    xi = linspace(mn, mx, ngrid)
    yi = linspace(mn, mx, ngrid)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata(points=(x, y), values=ln_2_err, xi=(xi, yi), method='linear')

    fig = plt.gcf()
    fig.gca().set_xlim(mn, mx)
    fig.gca().set_ylim(mn, mx)
    fig.gca().set_xlabel('r; C=0', fontsize=fontsize)
    fig.gca().set_ylabel('r; C=1', fontsize=fontsize)
    plt.tick_params(length=10, width=2, labelsize=fontsize)
    mpble = fig.gca().contourf(xi, yi, zi, levels, cmap=plt.cm.jet)
    cbar = plt.colorbar(mpble)
    cbar.set_label(r'log$_2$(err)', fontsize=fontsize, labelpad=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    for r in range(mx, 0, -1):
        patch = PathPatch(
            Path(
                [(0, r), (r * 4 * (sqrt(2) - 1) / 3, r), (r, r * 4 * (sqrt(2) - 1) / 3), (r, 0)],
                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            ),
            color='white',
            fill=False,
            linewidth=1,
            zorder=1
        )
        fig.gca().add_patch(patch)
    for i in range(len(cour)):
        c = cos(theta[i])
        s = sin(theta[i])
        fig.gca().add_patch(
            PathPatch(
                Path(
                    [(1 * c, 1 * s), (mx * c, mx * s)],
                    [Path.MOVETO, Path.LINETO]
                ),
                color='white',
                fill=None,
                linewidth=1
            )
        )
    fig.gca().contour(xi, yi, zi, levels, linewidths=1, colors='k')

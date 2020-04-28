## @file
# @author Anna Jaruga <ajaruga@igf.fuw.edu.pl>
# @copyright University of Warsaw
# @date Januar 2012
# @section LICENSE
# GPLv3+ (see the COPYING file or http://www.gnu.org/licenses/)

# 1d isolines test from pks & wwg 1990

# for plotting in detached screen
import matplotlib
import numpy as np
from numpy import loadtxt, zeros, linspace
from math import pi, log, cos, sin, sqrt
from scipy.interpolate import griddata
from matplotlib.patches import Path, PathPatch

import matplotlib.pyplot as plt

def plot(nr, cour, ln_2_err):

  theta = zeros(nr.shape[0])
  r = zeros(nr.shape[0])
  x = zeros(nr.shape[0])
  y = zeros(nr.shape[0])

  norm=max(nr)
  theta = cour * pi / 2.
  r = np.log(1/nr)
  # r += nr.shape[0]
  r -= min(r)
  r *= 10/max(r)
  for i in range(theta.shape[0]) :
    x[i] = r[i] * cos(theta[i])
    y[i] = r[i] * sin(theta[i])

  ngrid = 800 * 2
  n_levels = 10
  levels = np.linspace(min(ln_2_err), max(ln_2_err), n_levels)
  mn=0
  mx= int(np.ceil(max(r)))

  xi = linspace(mn, mx, ngrid)
  yi = linspace(mn, mx, ngrid)
  xi, yi = np.meshgrid(xi, yi)
  zi = griddata(points = (x,y), values = ln_2_err, xi = (xi, yi), method='linear')
  print(zi.shape)

  fig = plt.gcf()
  fig.gca().set_xlim(mn,mx)
  fig.gca().set_ylim(mn,mx)
  fig.gca().set_xlabel('r; C=0', fontsize = 30)
  fig.gca().set_ylabel('r; C=1', fontsize = 30)
  plt.tick_params(length=10, width=2, labelsize=24)


  mpble = fig.gca().contourf(xi,yi,zi,levels,cmap=plt.cm.jet)
  cbar = plt.colorbar(mpble)
  cbar.set_label(r'log$_2$(err)', fontsize=30, labelpad = 20)
  cbar.ax.tick_params(labelsize=24)
  for r in range(mx, 0, -1):
    zix = 1
    if (r==1): zix=10
    patch=PathPatch(
      Path(
        [(0,r),       (r*4*(sqrt(2)-1)/3,r), (r,r*4*(sqrt(2)-1)/3), (r,0)      ],
        [Path.MOVETO, Path.CURVE4,           Path.CURVE4,           Path.CURVE4]
      ),
      color='white',
      fill=(r==1),
      linewidth=1,
      zorder=zix
    )
    if (r!=1): fig.gca().add_patch(patch)
  for i in range(len(cour)) :
    c = cos(theta[i])
    s = sin(theta[i])
    fig.gca().add_patch(
      PathPatch(
        Path(
      [(1*c,1*s),   (mx*c,mx*s)  ],
      [Path.MOVETO, Path.LINETO]
        ),
        color='white',
        fill=None,
        linewidth=1
      )
    )
  fig.gca().contour(xi,yi,zi,levels,linewidths=1,colors='k')
  fig.gca().add_patch(patch)
  fig.show()

n = 12
if __name__ == '__main__':
    plot(np.random.random(n),np.random.random(n),np.random.random(n))
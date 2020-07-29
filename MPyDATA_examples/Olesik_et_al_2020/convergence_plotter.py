import numpy as np
from math import pi
import matplotlib.pyplot as plt


def polar_plot(nr, cour, values, name):
    theta_array = cour * pi / 2.
    dr = 1 / nr
    r_array = np.log2(dr)
    r_array -= (min(r_array) - 1)

    X, Y = np.meshgrid(theta_array, r_array)
    Z = np.array(list(values)).reshape(len(r_array), len(theta_array))

    min_val = np.floor(min(values)) if name == 'log$_2$(err)' else min(values)
    max_val = np.ceil(max(values)) if name == 'log$_2$(err)' else max(values)

    amplitude = max_val - min_val
    if name == 'log$_2$(err)':
        levels = np.linspace(
            min_val,
            max_val,
            int(amplitude + 1)
        )
    else:
        levels = 7

    ax = plt.subplot(111, projection='polar')
    cnt = ax.contourf(X, Y, Z, levels, cmap='jet')
    plt.contour(X, Y, Z, levels, colors='black', linewidth=.3)
    ax.scatter(X, Y, alpha=.2, s=10)
    legend = plt.colorbar(cnt, ax=ax, pad=0.1)

    ax.set_thetalim(0, np.pi/2)
    ax.set_rlim(0, max(r_array))
    theta_ticks = np.linspace(0, 90, 11)
    ax.set_thetagrids(theta_ticks, theta_ticks / 90)
    ax.grid(True)

    ax.set_title(name+" vs nr and C", va='bottom')

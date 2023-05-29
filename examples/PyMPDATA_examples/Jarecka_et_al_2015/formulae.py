# adapted from:
# https://github.com/igfuw/shallow-water-elliptic-drop/blob/master/analytical/analytic_equations.py
# Code used in the paper of Jarecka, Jaruga, Smolarkiewicz -
# "A Spreading Drop of Shallow Water" (JCP 289, doi:10.1016/j.jcp.2015.02.003).

import numpy as np
import numba
from scipy.integrate import odeint


def amplitude(x, y, lx, ly):
    A = 1 / lx / ly
    h = A * (1 - (x / lx) ** 2 - (y / ly) ** 2)
    return np.where(h > 0, h, 0)

@numba.njit()
def deriv(y, _):
    """
    return derivatives of [lambda_x, dlambda_x/dt, lambda_y, dlambda_y/dt
    four first-order ODEs based on  eq. 7  (Jarecka, Jaruga, Smolarkiewicz)
    """
    return np.array((y[1], 2. / y[0] ** 2 / y[2], y[3], 2. / y[0] / y[2] ** 2))


def d2_el_lamb_lamb_t_evol(times, lamb_x0, lamb_y0):
    """
    solving coupled nonlinear second-order ODEs - eq. 7  (Jarecka, Jaruga, Smolarkiewicz)
    returning array with first dim denoting time, second dim:
    [lambda_x, dot{lambda_x}, lambda_y, dot{lambda_y}
    """
    assert times[0] == 0
    yinit = np.array([lamb_x0, 0., lamb_y0, 0.])  # initial values (dot_lamb = 0.)
    result, info = odeint(deriv, yinit, times, full_output=True)
    assert info['message'] == 'Integration successful.'
    return result

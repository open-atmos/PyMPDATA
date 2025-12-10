# adapted from:
# https://github.com/igfuw/shallow-water-elliptic-drop/blob/master/analytical/analytic_equations.py
# Code used in the paper of Jarecka, Jaruga, Smolarkiewicz -
# "A Spreading Drop of Shallow Water" (JCP 289, doi:10.1016/j.jcp.2015.02.003).

import numba
import numpy as np
from scipy.integrate import odeint

from PyMPDATA.impl.enumerations import ARG_DATA, ARG_FOCUS, MAX_DIM_NUM


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
    return np.array((y[1], 2.0 / y[0] ** 2 / y[2], y[3], 2.0 / y[0] / y[2] ** 2))


def d2_el_lamb_lamb_t_evol(times, lamb_x0, lamb_y0):
    """
    solving coupled nonlinear second-order ODEs - eq. 7  (Jarecka, Jaruga, Smolarkiewicz)
    returning array with first dim denoting time, second dim:
    [lambda_x, dot{lambda_x}, lambda_y, dot{lambda_y}
    """
    assert times[0] == 0
    yinit = np.array([lamb_x0, 0.0, lamb_y0, 0.0])  # initial values (dot_lamb = 0.)
    result, info = odeint(deriv, yinit, times, full_output=True)
    assert info["message"] == "Integration successful."
    return result


def make_rhs_indexers(ats, grid_step, time_step, options):
    @numba.njit(**options.jit_flags)
    def rhs(m, _0, h, _1, _2, _3):
        retval = (
            m
            - ((ats(*h, +1) - ats(*h, -1)) / 2) / 2 * ats(*h, 0) * time_step / grid_step
        )
        return retval

    return rhs


def make_rhs(grid_step, time_step, axis, options, traversals):
    indexers = traversals.indexers[traversals.n_dims]
    apply_scalar = traversals.apply_scalar(loop=False)

    formulae_rhs = tuple(
        (
            make_rhs_indexers(
                ats=indexers.ats[axis],
                grid_step=grid_step[axis],
                time_step=time_step,
                options=options,
            ),
            None,
            None,
        )
    )

    @numba.njit(**options.jit_flags)
    def apply(traversals_data, momentum, h):
        null_scalarfield, null_scalarfield_bc = traversals_data.null_scalar_field
        null_vectorfield, null_vectorfield_bc = traversals_data.null_vector_field
        return apply_scalar(
            *formulae_rhs,
            *momentum.field,
            *null_vectorfield,
            null_vectorfield_bc,
            *h.field,
            h.bc,
            *null_scalarfield,
            null_scalarfield_bc,
            *null_scalarfield,
            null_scalarfield_bc,
            *null_scalarfield,
            null_scalarfield_bc,
            traversals_data.buffer
        )

    return apply


def make_interpolate_indexers(ati, options):
    @numba.njit(**options.jit_flags)
    def interpolate(momentum_x, _, momentum_y):
        momenta = (momentum_x[ARG_FOCUS], (momentum_x[ARG_DATA], momentum_y[ARG_DATA]))
        return ati(*momenta, 0.5)

    return interpolate


def make_interpolate(options, traversals):
    indexers = traversals.indexers[traversals.n_dims]
    apply_vector = traversals.apply_vector()

    formulae_interpolate = tuple(
        (
            make_interpolate_indexers(ati=indexers.ati[i], options=options)
            if indexers.ati[i] is not None
            else None
        )
        for i in range(MAX_DIM_NUM)
    )

    @numba.njit(**options.jit_flags)
    def apply(traversals_data, momentum_x, momentum_y, advector):
        null_vectorfield, null_vectorfield_bc = traversals_data.null_vector_field
        return apply_vector(
            *formulae_interpolate,
            *advector.field,
            *momentum_x.field,
            momentum_x.bc,
            *null_vectorfield,
            null_vectorfield_bc,
            *momentum_y.field,
            momentum_y.bc,
            traversals_data.buffer
        )

    return apply

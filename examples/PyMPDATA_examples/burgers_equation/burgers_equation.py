"""
Solution for the Burgers equation solution with MPDATA
"""

import numpy as np

from PyMPDATA import Options, ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Constant

OPTIONS = Options(nonoscillatory=False, infinite_gauge=True)


def initialize_simulation(nt, nx, t_max):
    """
    Initializes simulation variables and returns them.
    """
    dt = t_max / nt
    courants_x, dx = np.linspace(-1, 1, nx + 1, endpoint=True, retstep=True)
    x = courants_x[:-1] + dx / 2
    u0 = -np.sin(np.pi * x)

    stepper = Stepper(options=OPTIONS, n_dims=1)
    advectee = ScalarField(
        data=u0, halo=OPTIONS.n_halo, boundary_conditions=(Constant(0), Constant(0))
    )
    advector = VectorField(
        data=(np.full(courants_x.shape, 0.0),),
        halo=OPTIONS.n_halo,
        boundary_conditions=(Constant(0), Constant(0)),
    )
    solver = Solver(stepper=stepper, advectee=advectee, advector=advector)
    return dt, dx, x, advectee, advector, solver


def update_advector_n(vel, dt, dx, slice_idx):
    """
    Computes and returns the updated advector_n.
    """
    indices = np.arange(slice_idx.start, slice_idx.stop)
    return 0.5 * ((vel[indices] - vel[indices - 1]) / 2 + vel[:-1]) * dt / dx


def run_numerical_simulation(*, nt, nx, t_max):
    """
    Runs the numerical simulation and returns (states, x, dt, dx).
    """
    dt, dx, x, advectee, advector, solver = initialize_simulation(nt, nx, t_max)
    states = []
    vel = advectee.get()
    advector_n_1 = 0.5 * (vel[:-1] + np.diff(vel) / 2) * dt / dx
    assert np.all(advector_n_1 <= 1)
    i = slice(1, len(vel))

    for _ in range(nt):
        vel = advectee.get()
        advector_n = update_advector_n(vel, dt, dx, i)
        advector.get_component(0)[1:-1] = 0.5 * (3 * advector_n - advector_n_1)
        assert np.all(advector.get_component(0) <= 1)

        solver.advance(n_steps=1)
        advector_n_1 = advector_n.copy()
        states.append(solver.advectee.get().copy())

    return np.array(states), x, dt, dx

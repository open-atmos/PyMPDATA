# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
from PyMPDATA import Solver, Stepper, ScalarField, Options, VectorField
from PyMPDATA.boundary_conditions import Periodic


def test_upwind_1d():
    state = np.array([0, 1, 0])
    courant = 1

    options = Options(n_iters=1)
    mpdata = Solver(
        stepper=Stepper(options=options, n_dims=len(state.shape), non_unit_g_factor=False),
        advectee=ScalarField(
            state.astype(options.dtype),
            halo=options.n_halo,
            boundary_conditions=(Periodic(),)
        ),
        advector=VectorField(
            (np.full(state.shape[0] + 1, courant, dtype=options.dtype),),
            halo=options.n_halo,
            boundary_conditions=(Periodic(),)
        )
    )
    n_steps = 5

    conserved = np.sum(mpdata.advectee.get())
    mpdata.advance(n_steps)

    assert np.sum(mpdata.advectee.get()) == conserved

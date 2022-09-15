# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PyMPDATA import Options, ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Periodic


@pytest.mark.parametrize("n_iters", [2, 3, 4])
def test_double_pass_donor_cell(n_iters):
    courant = 0.5

    options = Options(n_iters=n_iters, DPDC=True, nonoscillatory=True)
    state = np.array([0, 1, 0], dtype=options.dtype)
    boundary_conditions = (Periodic(),)

    mpdata = Solver(
        stepper=Stepper(options=options, n_dims=state.ndim, non_unit_g_factor=False),
        advectee=ScalarField(
            state, halo=options.n_halo, boundary_conditions=boundary_conditions
        ),
        advector=VectorField(
            (np.full(state.shape[0] + 1, courant, dtype=options.dtype),),
            halo=options.n_halo,
            boundary_conditions=boundary_conditions,
        ),
    )
    steps = 1

    conserved = np.sum(mpdata.advectee.get())
    mpdata.advance(steps)

    assert np.sum(mpdata.advectee.get()) == conserved

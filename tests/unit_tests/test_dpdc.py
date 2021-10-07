import numpy as np
import pytest
from PyMPDATA import Solver, Stepper, ScalarField, Options, VectorField
from PyMPDATA.boundary_conditions import Periodic


@pytest.mark.parametrize("n_iters", [2, 3, 4])
def test_DPDC(n_iters):
    state = np.array([0, 1, 0])
    C = .5

    options = Options(n_iters=n_iters, DPDC=True, nonoscillatory=True)
    mpdata = Solver(
        stepper=Stepper(options=options, n_dims=len(state.shape), non_unit_g_factor=False),
        advectee=ScalarField(state.astype(options.dtype), halo=options.n_halo,
                             boundary_conditions=(Periodic(),)),
        advector=VectorField((np.full(state.shape[0] + 1, C, dtype=options.dtype),), halo=options.n_halo,
                             boundary_conditions=(Periodic(),))
    )
    nt = 1

    conserved = np.sum(mpdata.advectee.get())
    mpdata.advance(nt)

    assert np.sum(mpdata.advectee.get()) == conserved

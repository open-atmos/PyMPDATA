from PyMPDATA import Solver, Stepper, ScalarField, PeriodicBoundaryCondition, VectorField
from PyMPDATA.options import Options
import numpy as np


def test_upwind_1d():
    state = np.array([0, 1, 0])
    C = 1

    options = Options(n_iters=1)
    mpdata = Solver(
        stepper=Stepper(options=options, n_dims=len(state.shape), non_unit_g_factor=False),
        advectee=ScalarField(state.astype(options.dtype), halo=options.n_halo,
                             boundary_conditions=(PeriodicBoundaryCondition(),)),
        advector=VectorField((np.full(state.shape[0] + 1, C, dtype=options.dtype),), halo=options.n_halo,
                             boundary_conditions=(PeriodicBoundaryCondition(),))
    )
    nt = 5

    conserved = np.sum(mpdata.advectee.get())
    mpdata.advance(nt)

    assert np.sum(mpdata.advectee.get()) == conserved

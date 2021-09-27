import numpy as np
import pytest
from PyMPDATA import Solver, Stepper, ScalarField, PeriodicBoundaryCondition, VectorField
from PyMPDATA.options import Options


@pytest.mark.parametrize("n_iters", [2, 3, 4])
def test_DPDC(n_iters):
    state = np.array([0, 1, 0])
    C = .5

    options = Options(n_iters=n_iters, DPDC=True, flux_corrected_transport=True)
    mpdata = Solver(
        stepper=Stepper(options=options, n_dims=len(state.shape), non_unit_g_factor=False),
        advectee=ScalarField(state.astype(options.dtype), halo=options.n_halo,
                             boundary_conditions=(PeriodicBoundaryCondition(),)),
        advector=VectorField((np.full(state.shape[0] + 1, C, dtype=options.dtype),), halo=options.n_halo,
                             boundary_conditions=(PeriodicBoundaryCondition(),))
    )
    nt = 1

    conserved = np.sum(mpdata.advectee.get())
    mpdata.advance(nt)

    assert np.sum(mpdata.advectee.get()) == conserved

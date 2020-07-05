from MPyDATA.arakawa_c.scalar_field import ScalarField
from MPyDATA.arakawa_c.vector_field import VectorField
from MPyDATA.arakawa_c.boundary_condition.periodic_boundary_condition import PeriodicBoundaryCondition
from MPyDATA.factories import Stepper
from MPyDATA.options import Options
from MPyDATA.solver import Solver
import pytest
import numpy as np


@pytest.mark.parametrize("shape, ij0, out, C", [
    pytest.param((3, 1), (1, 0), np.array([[0.], [0.], [44.]]), (1., 0.)),
    pytest.param((1, 3), (0, 1), np.array([[0., 0., 44.]]), (0., 1.)),
    pytest.param((1, 3), (0, 1), np.array([[44., 0., 0.]]), (0., -1.)),
    pytest.param((3, 3), (1, 1), np.array([[0, 0, 0], [0, 0, 22], [0., 22., 0.]]), (.5, .5)),
    pytest.param((3, 3), (1, 1), np.array([[0, 0, 0], [0, 0, 22], [0., 22., 0.]]), (.5, .5)),
])
def test_upwind(shape, ij0, out, C):
    value = 44
    scalar_field_init = np.zeros(shape)
    scalar_field_init[ij0] = value

    vector_field_init = (
        np.full((shape[0] + 1, shape[1]), C[0]),
        np.full((shape[0], shape[1] + 1), C[1])
    )
    options = Options(n_iters=1)

    state = ScalarField(scalar_field_init, halo=options.n_halo, boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition()))
    GC_field = VectorField(vector_field_init, halo=options.n_halo, boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition()))

    mpdata = Solver(stepper=Stepper(options=options, grid=shape), advector=GC_field, advectee=state)
    mpdata.advance(1)

    np.testing.assert_array_equal(
        mpdata.advectee.get(),
        out
    )

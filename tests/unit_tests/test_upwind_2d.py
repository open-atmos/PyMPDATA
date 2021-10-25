# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import pytest
import numpy as np
from PyMPDATA import Options, Solver, Stepper, ScalarField, VectorField
from PyMPDATA.boundary_conditions import Periodic


@pytest.mark.parametrize("shape, ij0, out, courant_number", [
    pytest.param((3, 1), (1, 0), np.array([[0.], [0.], [44.]]), (1., 0.)),
    pytest.param((1, 3), (0, 1), np.array([[0., 0., 44.]]), (0., 1.)),
    pytest.param((1, 3), (0, 1), np.array([[44., 0., 0.]]), (0., -1.)),
    pytest.param((3, 3), (1, 1), np.array([[0, 0, 0], [0, 0, 22], [0., 22., 0.]]), (.5, .5)),
    pytest.param((3, 3), (1, 1), np.array([[0, 0, 0], [0, 0, 22], [0., 22., 0.]]), (.5, .5)),
])
def test_upwind(shape, ij0, out, courant_number):
    value = 44
    scalar_field_init = np.zeros(shape)
    scalar_field_init[ij0] = value

    vector_field_init = (
        np.full((shape[0] + 1, shape[1]), courant_number[0]),
        np.full((shape[0], shape[1] + 1), courant_number[1])
    )
    options = Options(n_iters=1)

    bcs = (Periodic(), Periodic())
    advectee = ScalarField(scalar_field_init, halo=options.n_halo, boundary_conditions=bcs)
    advector = VectorField(vector_field_init, halo=options.n_halo, boundary_conditions=bcs)

    mpdata = Solver(
        stepper=Stepper(options=options, grid=shape, n_threads=1),
        advector=advector,
        advectee=advectee
    )
    mpdata.advance(nt=1)

    np.testing.assert_array_equal(
        mpdata.advectee.get(),
        out
    )

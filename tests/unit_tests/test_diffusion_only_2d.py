from PyMPDATA import ScalarField, VectorField, PeriodicBoundaryCondition, Options, Solver, Stepper
import numpy as np


def test_diffusion_only_2d(
        data0=np.array([[0, 0, 0], [0, 1., 0], [0, 0, 0]]),
        mu=(.1, .1),
        nt=1
):
    # Arrange
    options = Options(non_zero_mu_coeff=True)
    bc = [PeriodicBoundaryCondition()] * 2
    advectee = ScalarField(data0, options.n_halo, bc)
    advector = VectorField(
        data=(
            np.zeros((data0.shape[0]+1, data0.shape[1])),
            np.zeros((data0.shape[0], data0.shape[1]+1))
        ),
        halo=options.n_halo, boundary_conditions=bc)
    solver = Solver(stepper=Stepper(options=options, grid=data0.shape), advector=advector, advectee=advectee)

    # Act
    solver.advance(nt=nt, mu_coeff=mu)

    # Assert
    data1 = solver.advectee.get()
    np.testing.assert_almost_equal(
        actual=np.sum(data1),
        desired=np.sum(data0)
    )
    assert np.amax(data0) > np.amax(data1)
    assert np.amin(data1) >= 0
    assert np.count_nonzero(data1) == 5

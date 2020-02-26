from MPyDATA_examples.condensational_growth.simulation import Simulation
from MPyDATA_examples.condensational_growth.setup import Setup
from MPyDATA_examples.condensational_growth.coord import x_id, x_ln, x_p2
from MPyDATA_examples.condensational_growth.analysis import figure_data
from MPyDATA.options import Options
import pytest
import numpy as np


@pytest.fixture(scope='module')
def data():
    return figure_data(debug=False)[0]  # TODO!


@pytest.mark.parametrize("coord", [x_id(), x_p2(), x_ln()])
@pytest.mark.parametrize("fct", [True, False])
def test_init(coord, fct):
    # Arrange
    opts = Options(nug=True, fct=fct)
    setup = Setup()
    # Act
    simulation = Simulation(coord, opts)
    simulation.solver.arrays.G.fill_halos()

    # Asserts for array shapes
    assert simulation.n.shape[0] == setup.nr

    # Asserts for Jacobian
    G_with_halo = simulation.solver.arrays.G._impl.data
    assert np.isfinite(G_with_halo).all()
    assert (np.diff(G_with_halo) >= 0).all() or (np.diff(G_with_halo) <= 0).all()


@pytest.mark.parametrize("coord", ['x_id', 'x_p2', 'x_ln'])
@pytest.mark.parametrize("opts", [
    "{'n_iters': 1}",
    "{'n_iters': 2, 'fct': True}",
    "{'n_iters': 3, 'dfl': True}",
    "{'n_iters': 2, 'tot': True, 'iga': True, 'fct': True}"
])
def test_n_finite(coord, opts, data):
    # Arrange
    psi = data[coord]['numerical'][opts][-1].magnitude

    # Assert
    assert np.isfinite(psi).all()
    assert 70 < np.amax(psi) < 225


@pytest.mark.parametrize("coord", ['x_id', 'x_p2', 'x_ln'])
@pytest.mark.parametrize("opts", [
    "{'n_iters': 1}",
    "{'n_iters': 2, 'fct': True}",
    "{'n_iters': 3, 'dfl': True}",
    "{'n_iters': 2, 'tot': True, 'iga': True, 'fct': True}"
])
def test_L2_finite(coord, opts, data):
    # Arrange
    sut = data[coord]['error_L2'][opts]

    # Assert
    assert np.isfinite(sut).all()


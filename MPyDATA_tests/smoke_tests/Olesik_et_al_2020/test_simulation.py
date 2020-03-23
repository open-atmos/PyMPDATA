from MPyDATA_examples.Olesik_et_al_2020.simulation import Simulation
from MPyDATA_examples.Olesik_et_al_2020.setup import Setup
from MPyDATA_examples.Olesik_et_al_2020.coordinates import x_id, x_ln, x_p2
from MPyDATA_examples.Olesik_et_al_2020.analysis import compute_figure_data
from MPyDATA.options import Options
import pytest
import numpy as np


@pytest.fixture(scope='module')
def data():
    return compute_figure_data()[0]  # TODO!


@pytest.mark.parametrize("psi_coord", [x_id(), x_p2(), x_ln()])
@pytest.mark.parametrize("grid_layout", [x_id(), x_p2(), x_ln()])
@pytest.mark.parametrize("fct", [False])  # TODO: check True
@pytest.mark.skip()
def test_init(grid_layout, psi_coord, fct):
    # Arrange
    opts = Options()#fct=fct)
    setup = Setup()
    # Act
    simulation = Simulation(setup, grid_layout=grid_layout, psi_coord=psi_coord, opts=opts)
    # simulation.solver.g_factor.fill_halos()

    # Asserts for array shapes
    assert simulation.n.shape[0] == setup.nr

    # Asserts for Jacobian
    G_with_halo = simulation.solver.g_factor.data
    assert np.isfinite(G_with_halo).all()
    assert (np.diff(G_with_halo) >= 0).all() or (np.diff(G_with_halo) <= 0).all()


@pytest.mark.parametrize("coord", ['x_id', 'x_p2', 'x_ln'])
@pytest.mark.parametrize("opts", [
    "{'n_iters': 1}",
    "{'n_iters': 2, 'fct': True}",
    "{'n_iters': 3, 'dfl': True}",
    "{'n_iters': 2, 'tot': True, 'iga': True, 'fct': True}"
])
@pytest.mark.skip()
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
@pytest.mark.skip()
def test_L2_finite(coord, opts, data):
    # Arrange
    sut = data[coord]['error_L2'][opts]

    # Assert
    assert np.isfinite(sut).all()

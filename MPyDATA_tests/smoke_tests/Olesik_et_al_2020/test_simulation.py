from MPyDATA_examples.Olesik_et_al_2020.simulation import Simulation
from MPyDATA_examples.Olesik_et_al_2020.setup import Setup, default_dt, default_nr
from MPyDATA_examples.Olesik_et_al_2020.coordinates import x_id, x_log_of_p3, x_p2, x_p3, x_ln
from MPyDATA_examples.Olesik_et_al_2020.analysis import compute_figure_data
from MPyDATA.options import Options
import pytest
import numpy as np

grid_layout_set = (x_id(), x_p2(), x_ln())
opt_set = (
    {'n_iters': 1},
    {'n_iters': 2, 'flux_corrected_transport': True},
    {'n_iters': 3, 'third_order_terms': True, 'infinite_gauge': True, 'flux_corrected_transport': True})

@pytest.fixture(scope='module')
def data():
    result, _ =  compute_figure_data(nr = default_nr , dt = default_dt, psi_coord=x_id(),
                        grid_layouts=grid_layout_set,
                        opt_set=opt_set)
    return result


@pytest.mark.parametrize("psi_coord", [x_id(), x_p2(), x_p3()])
@pytest.mark.parametrize("grid_layout", [x_id(), x_p2(),  x_log_of_p3()])
@pytest.mark.parametrize("flux_corrected_transport", [False, True])
@pytest.mark.skip()
def test_init(grid_layout, psi_coord, flux_corrected_transport):
    # Arrange
    opts = Options(flux_corrected_transport = flux_corrected_transport)
    setup = Setup()
    # Act
    simulation = Simulation(setup, grid_layout=grid_layout, psi_coord=psi_coord, opts=opts)
    simulation.step(1)
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


@pytest.mark.parametrize("grid_layout", grid_layout_set)
@pytest.mark.parametrize("opts", opt_set)
def test_L2_finite(grid_layout, opts, data):
    # Arrange
    grid_layout_str = grid_layout.__class__.__name__
    sut = data[grid_layout_str]['error_L2'][str(opts)]

    # Assert
    assert np.isfinite(sut).all()


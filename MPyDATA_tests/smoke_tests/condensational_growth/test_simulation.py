from MPyDATA_examples.condensational_growth.simulation import Simulation
from MPyDATA_examples.condensational_growth.setup import Setup
from MPyDATA_examples.condensational_growth.coord import x_id, x_ln, x_p2
from MPyDATA.options import Options
import pint, pytest
import numpy as np


@pytest.mark.parametrize("coord", [x_id(), x_p2(), x_ln()])
@pytest.mark.parametrize("fct", [True, False])
def test_init(coord, fct):
    # Arrange
    si = pint.UnitRegistry()
    setup = Setup(si)
    opts = Options(nug=True, fct=fct)

    # Act
    simulation = Simulation(coord, setup, opts)
    simulation.solver.arrays.G.fill_halos()

    # Asserts for array shapes
    assert simulation.n.shape[0] == setup.nr

    # Asserts for Jacobian
    G_with_halo = simulation.solver.arrays.G._impl.data
    assert np.isfinite(G_with_halo).all()
    assert (np.diff(G_with_halo) >= 0).all() or (np.diff(G_with_halo) < 0).all()


@pytest.mark.parametrize("coord", [x_id(), x_p2(), x_ln()])
def test_step(coord, n_iters=1):
    # Arrange
    si = pint.UnitRegistry()
    setup = Setup(si)
    opts = Options(nug=True)
    simulation = Simulation(coord, setup, opts)
    G = simulation.solver.arrays.G.get()

    # Act
    # for step in range(setup.nt[-1]):
    for step in range(100):
        psi = simulation.solver.arrays.curr.get()
        Gpsi_sum0 = np.sum(G * psi)
        simulation.step(n_iters=n_iters, check_conservativeness=True)
        psi = simulation.solver.arrays.curr.get()

        bcflux = 0
        flux = simulation.solver.arrays.GC_prev  # TODO...
        flux._impl.focus(setup.nr-1)
        bcflux += flux._impl.at(+.5, 0)
        flux._impl.focus(0)
        bcflux += flux._impl.at(-.5, 0)
        print(bcflux)
        Gpsi_sum1 = np.sum(G * psi)
        np.testing.assert_approx_equal(
            desired=Gpsi_sum0,
            actual=(Gpsi_sum1 + bcflux),
            significant=13
        )

    # Assert
    assert np.isfinite(simulation.n).all()

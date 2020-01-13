from MPyDATA_examples.condensational_growth.simulation import Simulation
from MPyDATA_examples.condensational_growth.setup import Setup
from MPyDATA_examples.condensational_growth.coord import x_id, x_ln, x_p2
from MPyDATA.options import Options
import pint, pytest
import numpy as np


@pytest.mark.parametrize("coord", [x_id(), x_p2(), x_ln()])
def test_init(coord):
    # Arrange
    si = pint.UnitRegistry()
    setup = Setup(si)
    opts = Options(nug=True)

    # Act
    simulation = Simulation(coord, setup, opts)
    simulation.solver.arrays.G.fill_halos()

    # Assert
    G_with_halo = simulation.solver.arrays.G._impl.data
    assert np.isfinite(G_with_halo).all()
    assert (np.diff(G_with_halo) >= 0).all() or (np.diff(G_with_halo) < 0).all()

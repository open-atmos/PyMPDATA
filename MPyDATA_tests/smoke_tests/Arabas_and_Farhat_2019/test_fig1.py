from MPyDATA_examples.Arabas_and_Farhat_2019.simulation import Simulation
from MPyDATA_examples.Arabas_and_Farhat_2019.setup1_european_corridor import Setup
import numpy as np
import pytest


@pytest.fixture(scope="module")
def setup():
    return Setup()


@pytest.fixture(scope="module")
def simulation(setup):
    return Simulation(setup)


@pytest.fixture(scope="module")
def psi_0(simulation):
    return simulation.run(n_iters=2)


@pytest.fixture(scope="module")
def psi_a(simulation, setup):
    return setup.analytical_solution(simulation.S)


class TestFig1:
    @staticmethod
    def test_psi0_max(psi_0, setup):
        scl = setup.K2 - setup.K1
        np.testing.assert_almost_equal(np.amax(psi_0 / scl), 1, decimal=2)

    @staticmethod
    def test_psi0_min(psi_0, setup):
        scl = setup.K2 - setup.K1
        np.testing.assert_almost_equal(np.amin(psi_0 / scl), 0, decimal=14)

    @staticmethod
    @pytest.mark.skip()  # TODO
    def test_abserr(psi_0, psi_a):
        abserr = psi_0 - psi_a
        maxabserr = np.amax(np.abs(abserr))
        assert np.abs(abserr[0]) < .0001 * maxabserr
        assert np.abs(abserr[-1]) < .001 * maxabserr
        assert maxabserr < .75

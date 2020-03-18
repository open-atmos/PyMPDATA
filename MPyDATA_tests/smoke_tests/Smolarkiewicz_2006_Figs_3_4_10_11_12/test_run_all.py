from MPyDATA_examples.Smolarkiewicz_2006_Figs_3_4_10_11_12.simulation import Simulation
from MPyDATA_examples.Smolarkiewicz_2006_Figs_3_4_10_11_12.setup import Setup
from MPyDATA.options import Options
import numpy as np


class TestSmolarkiewicz2006:
    @staticmethod
    def test_fig3():
        # Arrange
        simulation = Simulation(Setup("cosine"), Options(n_iters=1))
        psi0 = simulation.state

        # Act
        simulation.run()
        psiT = simulation.state

        # Assert
        epsilon = 1e-20
        assert np.amin(psi0) == 0
        assert np.amax(psi0) == 2
        assert 0 < np.amin(psiT) < epsilon
        assert .45 < np.amax(psiT) < .5

    @staticmethod
    def test_fig4():
        # Arrange
        simulation = Simulation(Setup("cosine"), Options(n_iters=2))
        psi0 = simulation.state

        # Act
        simulation.run()
        psiT = simulation.state

        # Assert
        epsilon = 1e-20
        assert np.amin(psi0) == 0
        assert np.amax(psi0) == 2
        assert 0 < np.amin(psiT) < epsilon
        assert 1.3 < np.amax(psiT) < 1.4

    @staticmethod
    def test_fig10():
        # Arrange
        simulation = Simulation(Setup("cosine"), Options(infinite_gauge=True, n_iters=2))

        # Act
        simulation.run()
        psiT = simulation.state

        # Assert
        assert -.1 < np.amin(psiT) < 0
        assert 1.75 < np.amax(psiT) < 1.9


    @staticmethod
    def test_fig11():
        # Arrange
        simulation = Simulation(Setup("rect"), Options(infinite_gauge=True, n_iters=2))

        # Act
        simulation.run()
        psiT = simulation.state

        # Assert
        assert -1.9 < np.amin(psiT) < 2
        assert 4 < np.amax(psiT) < 4.2


    # TODO
    # @staticmethod
    # def test_fig12():
    #     # Arrange
    #     simulation = Simulation(Setup("rect"), Options(iga=True, fct=True), n_iters=2, debug=True)
    #
    #     # Act
    #     simulation.run()
    #     psiT = simulation.state
    #
    #     # Assert
    #     assert np.amin(psiT) >= 2
    #     assert np.amax(psiT) <= 4

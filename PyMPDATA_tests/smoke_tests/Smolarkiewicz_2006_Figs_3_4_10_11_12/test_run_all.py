from PyMPDATA_examples.Smolarkiewicz_2006_Figs_3_4_10_11_12.simulation import Simulation
from PyMPDATA_examples.Smolarkiewicz_2006_Figs_3_4_10_11_12.settings import Settings
from PyMPDATA.options import Options
import numpy as np
import pytest


class TestSmolarkiewicz2006:
    dtypes = (np.float32, np.float64)

    @staticmethod
    @pytest.mark.parametrize("dtype", dtypes)
    def test_fig3(dtype: np.floating):
        # Arrange
        simulation = Simulation(Settings("cosine"), Options(n_iters=1, dtype=dtype))
        psi0 = simulation.state

        # Act
        simulation.run()
        psiT = simulation.state

        # Assert
        epsilon = 1e-20
        assert psiT.dtype == dtype
        assert np.amin(psi0) == 0
        assert np.amax(psi0) == 2
        assert 0 < np.amin(psiT) < epsilon
        assert .45 < np.amax(psiT) < .5

    @staticmethod
    @pytest.mark.parametrize("dtype", dtypes)
    def test_fig4(dtype: np.floating):
        # Arrange
        simulation = Simulation(Settings("cosine"), Options(n_iters=2, dtype=dtype))
        psi0 = simulation.state

        # Act
        simulation.run()
        psiT = simulation.state

        # Assert
        epsilon = 1e-20
        assert psiT.dtype == dtype
        assert np.amin(psi0) == 0
        assert np.amax(psi0) == 2
        assert 0 < np.amin(psiT) < epsilon
        assert 1.3 < np.amax(psiT) < 1.4

    @staticmethod
    @pytest.mark.parametrize("dtype", dtypes)
    def test_fig10(dtype: np.floating):
        # Arrange
        simulation = Simulation(Settings("cosine"), Options(infinite_gauge=True, n_iters=2, dtype=dtype))

        # Act
        simulation.run()
        psiT = simulation.state

        # Assert
        assert psiT.dtype == dtype
        assert -.1 < np.amin(psiT) < 0
        assert 1.75 < np.amax(psiT) < 1.9

    @staticmethod
    @pytest.mark.parametrize("dtype", dtypes)
    def test_fig11(dtype: np.floating):
        # Arrange
        simulation = Simulation(Settings("rect"), Options(infinite_gauge=True, n_iters=2, dtype=dtype))

        # Act
        simulation.run()
        psiT = simulation.state

        # Assert
        assert psiT.dtype == dtype
        assert -1.9 < np.amin(psiT) < 2
        assert 4 < np.amax(psiT) < 4.2

    @staticmethod
    @pytest.mark.parametrize("dtype", dtypes)
    def test_fig12(dtype: np.floating):
        # Arrange
        simulation = Simulation(Settings("rect"), Options(n_iters=2, infinite_gauge=True, flux_corrected_transport=True, dtype=dtype))

        # Act
        simulation.run()
        psiT = simulation.state

        # Assert
        assert psiT.dtype == dtype
        assert np.amin(psiT) >= 2
        assert np.amax(psiT) <= 4
        assert np.amax(psiT) > 3

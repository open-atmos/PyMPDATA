# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from PyMPDATA_examples.Smolarkiewicz_2006_Figs_3_4_10_11_12.settings import Settings
from PyMPDATA_examples.Smolarkiewicz_2006_Figs_3_4_10_11_12.simulation import Simulation

from PyMPDATA.options import Options


class TestSmolarkiewicz2006:
    dtypes = (np.float32, np.float64)

    @staticmethod
    @pytest.mark.parametrize("dtype", dtypes)
    def test_fig3(dtype: np.floating):
        # Arrange
        simulation = Simulation(Settings("cosine"), Options(n_iters=1, dtype=dtype))
        psi_0 = simulation.state

        # Act
        simulation.run()
        psi_t = simulation.state

        # Assert
        epsilon = 1e-20
        assert psi_t.dtype == dtype
        assert np.amin(psi_0) == 0
        assert np.amax(psi_0) == 2
        assert 0 < np.amin(psi_t) < epsilon
        assert 0.45 < np.amax(psi_t) < 0.5

    @staticmethod
    @pytest.mark.parametrize("dtype", dtypes)
    def test_fig4(dtype: np.floating):
        # Arrange
        simulation = Simulation(Settings("cosine"), Options(n_iters=2, dtype=dtype))
        psi_0 = simulation.state

        # Act
        simulation.run()
        psi_t = simulation.state

        # Assert
        epsilon = 1e-20
        assert psi_t.dtype == dtype
        assert np.amin(psi_0) == 0
        assert np.amax(psi_0) == 2
        assert 0 < np.amin(psi_t) < epsilon
        assert 1.3 < np.amax(psi_t) < 1.4

    @staticmethod
    @pytest.mark.parametrize("dtype", dtypes)
    def test_fig10(dtype: np.floating):
        # Arrange
        simulation = Simulation(
            Settings("cosine"), Options(infinite_gauge=True, n_iters=2, dtype=dtype)
        )

        # Act
        simulation.run()
        psi_t = simulation.state

        # Assert
        assert psi_t.dtype == dtype
        assert -0.1 < np.amin(psi_t) < 0
        assert 1.75 < np.amax(psi_t) < 1.9

    @staticmethod
    @pytest.mark.parametrize("dtype", dtypes)
    def test_fig11(dtype: np.floating):
        # Arrange
        simulation = Simulation(
            Settings("rect"), Options(infinite_gauge=True, n_iters=2, dtype=dtype)
        )

        # Act
        simulation.run()
        psi_t = simulation.state

        # Assert
        assert psi_t.dtype == dtype
        assert -1.9 < np.amin(psi_t) < 2
        assert 4 < np.amax(psi_t) < 4.2

    @staticmethod
    @pytest.mark.parametrize("dtype", dtypes)
    def test_fig12(dtype: np.floating):
        # Arrange
        simulation = Simulation(
            Settings("rect"),
            Options(n_iters=2, infinite_gauge=True, nonoscillatory=True, dtype=dtype),
        )

        # Act
        simulation.run()
        psi_t = simulation.state

        # Assert
        assert psi_t.dtype == dtype
        assert np.amin(psi_t) >= 2
        assert np.amax(psi_t) <= 4
        assert np.amax(psi_t) > 3

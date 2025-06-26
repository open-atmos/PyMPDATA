"""Unit tests for the Burgers' equation numerical simulation."""

import numpy as np
import pytest
from PyMPDATA_examples.burgers_equation import run_numerical_simulation


@pytest.fixture(name="states")
def states_fixture():
    """Run the simulation once for all tests."""
    return run_numerical_simulation(nt=400, nx=100, t_max=1 / np.pi)[0]


class TestBurgersEquation:
    """Test suite for general numerical verification of Burgers' equation simulation."""

    @staticmethod
    def test_total_momentum_conservation(states):
        """Verify total momentum remains approximately constant over time."""
        sum_initial_state = np.sum(states[0])
        eps = 1e-5

        for state in states:
            sum_state = np.sum(state)
            np.testing.assert_allclose(
                desired=sum_initial_state,
                actual=sum_state,
                atol=eps,
            )

    @staticmethod
    def test_solution_within_bounds(states):
        """Ensure numerical solution u(i, j) stays within expected bounds."""
        eps = 1e-2
        min_val = np.min(states) + eps
        max_val = np.max(states) - eps
        assert min_val >= -1.0
        assert max_val <= 1.0

    @staticmethod
    def test_zero_constant_boundary_conditions(states):
        """Verify zero-constant boundary conditions are satisfied at all time steps."""
        eps = 5e-2

        for state in states:
            left_boundary = state[0]
            right_boundary = state[-1]

            np.testing.assert_allclose(
                desired=0,
                actual=left_boundary,
                atol=eps,
            )
            np.testing.assert_allclose(
                desired=0,
                actual=right_boundary,
                atol=eps,
            )

# pylint: disable=line-too-long
"""Unit tests for the Burgers' equation numerical simulation."""

import unittest
import numpy as np
from examples.PyMPDATA_examples.burgers_equation.burgers_equation import run_numerical_simulation


class TestGeneralNumericalVeryfication(unittest.TestCase):
    """Test suite for general numerical verification of Burgers' equation simulation."""

    def setUp(self):
        """Run the simulation once for all tests."""
        self.states, self.x, self.dt, self.dx = run_numerical_simulation(nt=400, nx=100, t_max=1/np.pi)

    def test_total_momentum_conservation(self):
        """Verify total momentum remains approximately constant over time."""
        sum_initial_state = np.sum(self.states[0])
        eps = 1e-5

        for state in self.states:
            sum_state = np.sum(state)
            self.assertAlmostEqual(sum_initial_state, sum_state, delta=eps,
                                   msg=f"Total momentum changed: {sum_initial_state} != {sum_state}")

    def test_solution_within_bounds(self):
        """Ensure numerical solution u(i, j) stays within expected bounds."""
        eps = 1e-2
        min_val = np.min(self.states) + eps
        max_val = np.max(self.states) - eps
        self.assertGreaterEqual(min_val, -1.0, f"Minimum value {min_val} is less than -1.")
        self.assertLessEqual(max_val, 1.0, f"Maximum value {max_val} is greater than 1.")

    def test_periodic_boundary_conditions(self):
        """Verify periodic boundary conditions are satisfied at all time steps."""
        eps = 1e-5

        for state in self.states:
            left_boundary = state[0]
            right_boundary = state[-1]

            self.assertAlmostEqual(0, left_boundary, delta=eps,
                        msg=f"Periodic boundary condition violated: 0 != {left_boundary}")
            self.assertAlmostEqual(0, right_boundary, delta=eps,
                                   msg=f"Periodic boundary condition violated: 0 != {right_boundary}")


if __name__ == '__main__':
    unittest.main()

"""
Test the similarity of solutions from MPDATA and PyPDE for 2D diffusion
"""

import numpy as np
import pytest
from numpy.ma.testutils import assert_almost_equal

from examples.PyMPDATA_examples.comparison_against_pypde_et_al_2025.diffusion_2d import (
    InitialConditions,
    Two2DiffusionSolution,
    mpdata_solution,
    py_pde_solution,
)


@pytest.fixture(name="initial_conditions")
def _initial_conditions() -> InitialConditions:
    """Fixture providing initial conditions for the diffusion problem."""

    return InitialConditions(
        diffusion_coefficient=0.1,
        time_step=0.001,
        time_end=2,
        grid_shape=(30, 18),
        grid_range_x=(-1.0, 1.0),
        grid_range_y=(0.0, 2.0),
        pulse_position=(0.0, 1.0),
    )


def test_similarity_of_solutions(initial_conditions: InitialConditions) -> None:
    """Test that the solutions from PyPDE and MPDATA for 2D diffusion are similar."""

    py_pde_result: Two2DiffusionSolution = py_pde_solution(
        initial_conditions=initial_conditions,
    )
    mpdata_result: Two2DiffusionSolution = mpdata_solution(
        initial_conditions=initial_conditions,
    )

    # sanity checks
    assert py_pde_result.shape == mpdata_result.shape
    assert np.all(np.isfinite(py_pde_result))
    assert np.all(np.isfinite(mpdata_result))
    assert np.all(py_pde_result >= 0)
    assert np.all(mpdata_result >= 0)

    # total mass check
    assert_almost_equal(py_pde_result.sum(), mpdata_result.sum(), decimal=5)

    # compare results
    corr = np.corrcoef(py_pde_result.ravel(), mpdata_result.ravel())[0, 1]
    diff = py_pde_result - mpdata_result
    rmse = np.sqrt(np.mean(diff**2))
    l1_error = np.sum(np.abs(diff)) * initial_conditions.dx * initial_conditions.dy
    l2_norm = np.linalg.norm(py_pde_result - mpdata_result)
    assert np.allclose(py_pde_result, mpdata_result, rtol=1e2, atol=1e3)
    assert corr > 0.97
    assert rmse < 0.2
    assert l1_error < 0.5
    assert l2_norm < 3.5

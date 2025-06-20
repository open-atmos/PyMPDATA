"""
Test the similarity of solutions from MPDATA and PyPDE for 2D diffusion
"""

import numpy as np
import pytest
import time

from examples.PyMPDATA_examples.comparison_against_pypde_2025.diffusion_2d import (
    InitialConditions,
    Grid,
    mpdata_solution,
    py_pde_solution,
)


@pytest.fixture(name="initial_conditions")
def _initial_conditions() -> InitialConditions:
    """Fixture providing initial conditions for the diffusion problem."""

    return InitialConditions(
        diffusion_coefficient=0.1,
        time_step=0.001,
        time_end=1,
        grid_shape=(30, 18),
        grid_range_x=(-1.0, 1.0),
        grid_range_y=(0.0, 2.0),
        pulse_position=(0.0, 1.0),
        pulse_shape=(2, 2),
    )


def test_initial_conditions(initial_conditions: InitialConditions) -> None:
    """Test that the initial conditions are set up correctly."""

    assert initial_conditions.diffusion_coefficient > 0
    assert initial_conditions.time_step > 0
    assert initial_conditions.time_end > 0
    assert isinstance(initial_conditions.grid_shape, tuple)
    assert len(initial_conditions.grid_shape) == 2
    assert all(isinstance(dim, int) for dim in initial_conditions.grid_shape)
    assert all(dim > 0 for dim in initial_conditions.grid_shape)
    assert isinstance(initial_conditions.grid_range_x, tuple)
    assert len(initial_conditions.grid_range_x) == 2
    assert isinstance(initial_conditions.grid_range_y, tuple)
    assert len(initial_conditions.grid_range_y) == 2
    assert isinstance(initial_conditions.pulse_position, tuple)
    assert len(initial_conditions.pulse_position) == 2
    assert isinstance(initial_conditions.pulse_shape, tuple)
    assert len(initial_conditions.pulse_shape) == 2


def test_similarity_of_solutions(initial_conditions: InitialConditions) -> None:
    """Test that the solutions from PyPDE and MPDATA for 2D diffusion are similar."""

    # initial solutions
    py_pde_result: Grid = py_pde_solution(
        initial_conditions=initial_conditions,
    )

    mpdata_result: Grid = mpdata_solution(
        initial_conditions=initial_conditions,
    )

    # calculate solutions again to time them and ensure they are consistent across runs
    py_pde_start = time.perf_counter()
    py_pde_result2: Grid = py_pde_solution(
        initial_conditions=initial_conditions,
    )
    assert np.all(py_pde_result == py_pde_result2), (
        "PyPDE results are not consistent across runs"
    )

    MAX_ELAPSED_TIME = 10  # seconds
    assert py_pde_result.shape == mpdata_result.shape
    py_pde_elapsed = time.perf_counter() - py_pde_start
    assert py_pde_elapsed < MAX_ELAPSED_TIME, "PyPDE solution took too long to compute"

    mpdata_start = time.perf_counter()
    mpdata_result2: Grid = mpdata_solution(
        initial_conditions=initial_conditions,
    )
    mpdata_elapsed = time.perf_counter() - mpdata_start
    assert mpdata_elapsed < MAX_ELAPSED_TIME, "MPDATA solution took too long to compute"

    assert np.all(mpdata_result == mpdata_result2), (
        "MPDATA results are not consistent across runs"
    )

    # sanity checks
    assert py_pde_result.shape == mpdata_result.shape
    assert np.all(np.isfinite(py_pde_result))
    assert np.all(np.isfinite(mpdata_result))
    assert np.all(py_pde_result >= 0)
    assert np.all(mpdata_result >= 0)

    # total mass check
    assert np.isclose(py_pde_result.sum(), mpdata_result.sum(), rtol=1e-5, atol=1e-10)

    py_pde_total_mass = (
        py_pde_result.sum() * initial_conditions.dx * initial_conditions.dy
    )
    mpdata_total_mass = (
        mpdata_result.sum() * initial_conditions.dx * initial_conditions.dy
    )
    assert np.isclose(py_pde_total_mass, 1.0, rtol=1e-3)
    assert np.isclose(mpdata_total_mass, 1.0, rtol=1e-3)

    # compare results
    corr = np.corrcoef(py_pde_result.ravel(), mpdata_result.ravel())[0, 1]
    diff = py_pde_result - mpdata_result
    l1_error = np.sum(np.abs(diff)) * initial_conditions.dx * initial_conditions.dy
    l2_norm = np.linalg.norm(diff)
    rmse = l2_norm / np.sqrt(py_pde_result.size)
    assert np.allclose(py_pde_result, mpdata_result, rtol=1e-2, atol=1e-1)
    assert corr > 0.99
    assert rmse < 0.02
    assert l1_error < 0.05
    assert l2_norm < 0.5

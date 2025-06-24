import numpy as np
import pde as py_pde
import pytest
from PyMPDATA_examples.comparisons_against_pypde.diffusion_equation_with_spatial_dependence import (
    solutions,
)

from PyMPDATA import Options


@pytest.fixture
def simulation_args() -> solutions.SimulationArgs:
    """Fixture with the simulation arguments."""

    return solutions.SimulationArgs(
        grid_bounds=(-5.0, 5.0),
        grid_points=64,
        initial_value=1.0,
        sim_time=100.0,
        dt=1e-3,
    )


def test_similarity(simulation_args: solutions.SimulationArgs):
    """Tests that the results of the two implementations (py-pde and PyMPDATA) are similar."""

    assert hasattr(
        Options, "heterogeneous_diffusion"
    ), "Options should have heterogeneous_diffusion field"

    py_pde_result = solutions.py_pde_solution(simulation_args)

    pympdata_result = solutions.pympdata_solution(simulation_args)

    difference = np.abs(
        pympdata_result.kymograph_result - py_pde_result.kymograph_result
    )
    rmse = np.sqrt(np.mean(difference**2))

    assert (
        pympdata_result.kymograph_result.shape == py_pde_result.kymograph_result.shape
    ), "Kymograph results from both implementations should have the same shape."
    assert np.allclose(
        pympdata_result.kymograph_result, py_pde_result.kymograph_result, atol=0.2
    ), "Kymograph results from both implementations should be similar within the tolerance."

    assert rmse < 0.05


def test_consistency_across_runs(simulation_args: solutions.SimulationArgs):
    """Tests that the results of the two implementations (py-pde and PyMPDATA) are similar."""

    assert hasattr(
        Options, "heterogeneous_diffusion"
    ), "Options should have heterogeneous_diffusion field"

    py_pde_result_1 = solutions.py_pde_solution(simulation_args)
    py_pde_result_2 = solutions.py_pde_solution(simulation_args)

    assert (
        py_pde_result_1.kymograph_result.shape == py_pde_result_2.kymograph_result.shape
    ), "Kymograph results from both runs should have the same shape."
    assert np.allclose(
        py_pde_result_1.kymograph_result,
        py_pde_result_2.kymograph_result,
        atol=0.2,
    ), "Kymograph results from both runs should be similar within the tolerance."

    pympdata_result_1 = solutions.pympdata_solution(simulation_args)
    pympdata_result_2 = solutions.pympdata_solution(simulation_args)

    assert (
        pympdata_result_1.kymograph_result.shape
        == pympdata_result_2.kymograph_result.shape
    ), "Kymograph results from both runs should have the same shape."
    assert np.allclose(
        pympdata_result_1.kymograph_result,
        pympdata_result_2.kymograph_result,
        atol=0.2,
    ), "Kymograph results from both runs should be similar within the tolerance."

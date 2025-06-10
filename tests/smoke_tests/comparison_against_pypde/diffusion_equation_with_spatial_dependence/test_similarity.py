import numpy as np
import pde as py_pde

from PyMPDATA_examples.comparisons_against_pypde.diffusion_equation_with_spatial_dependence import (
    solutions,
)
from PyMPDATA import Options


def test_similarity():
    """Tests that the results of the two implementations (py-pde and PyMPDATA) are similar."""

    assert hasattr(
        Options, "heterogeneous_diffusion"
    ), "Options should have heterogeneous_diffusion field"

    simulation_args = solutions.SimulationArgs(
        grid_bounds=(-5.0, 5.0),
        grid_points=64,
        initial_value=1.0,
        sim_time=100.0,
        dt=1e-3,
    )

    plot_path = "tests/smoke_tests/comparison_against_pypde/diffusion_equation_with_spatial_dependence/py_pde_kymograph.png"
    py_pde_result = solutions.py_pde_solution(simulation_args)

    py_pde.plot_kymograph(
        py_pde_result.extra["storage"], filename=plot_path, action="none"
    )

    plot_path = "tests/smoke_tests/comparison_against_pypde/diffusion_equation_with_spatial_dependence/pympdata_kymograph.png"
    pympdata_result = solutions.pympdata_solution(simulation_args)

    pympdata_result.figures["kymograph"].savefig(plot_path, dpi=300)

    assert (
        pympdata_result.kymograph_result.shape == py_pde_result.kymograph_result.shape
    ), "Kymograph results from both implementations should have the same shape."
    assert np.allclose(
        pympdata_result.kymograph_result, py_pde_result.kymograph_result, atol=5 * 1e-1
    ), "Kymograph results from both implementations should be similar within the tolerance."

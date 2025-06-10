import numpy as np
import pde as py_pde

import PyMPDATA_examples.comparisons_against_pypde.diffusion_equation_with_spatial_dependence.solutions as solutions


def test_similarity():
    """Test that the results of the two implementations (py-pde and PyMPDATA) are similar."""

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

    assert np.allclose(
        pympdata_result.kymograph_result, py_pde_result.kymograph_result, atol=1e-3
    )

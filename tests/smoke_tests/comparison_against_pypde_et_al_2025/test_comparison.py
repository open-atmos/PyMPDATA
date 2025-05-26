from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numpy.ma.testutils import assert_almost_equal

from examples.PyMPDATA_examples.comparison_against_pypde_et_al_2025.diffusion_2d import (
	py_pde_solution,
	mpdata_solution,
	analytical_solution,
)


@dataclass(frozen=True)
class Two2DiffusionSolution:
	result: npt.NDArray[np.float64]

	@property
	def total_mass(self) -> float:
		return self.result.sum()  # * dx * dy


def test_similarity_of_solutions() -> None:
	D = 0.1  # Diffusion coefficient
	dt = 0.001  # Time step size
	t_end = 2
	nx, ny = 30, 18
	min_x, max_x = -1.0, 1.0
	min_y, max_y = 0.0, 2.0

	dx = (max_x - min_x) / nx
	dy = (max_y - min_y) / ny

	py_pde: Two2DiffusionSolution = py_pde_solution()
	mpdata: Two2DiffusionSolution = mpdata_solution()

	# sanitiy checks
	assert py_pde.result.shape == mpdata.result.shape
	assert np.all(np.isfinite(py_pde))
	assert np.all(np.isfinite(solution_pypde))
	assert np.all(solution_pymp >= 0)
	assert np.all(solution_pypde >= 0)
	# initial condition must be the same
	assert np.allclose(initial_condition_pymp, initial_condition_pypde)
	# run the solver twice with the same seed to ensure reproducibility
	sol1 = run_solver(seed=42)
	sol2 = run_solver(seed=42)
	assert np.allclose(sol1, sol2)
	# run the solver twice with the same seed to ensure reproducibility
	sol1 = run_solver(seed=42)
	sol2 = run_solver(seed=42)
	assert np.allclose(sol1, sol2)

	# compare results
	assert np.allclose(py_pde.result, mpdata.result, rtol=1e-6, atol=1e-8)

	assert_almost_equal(py_pde.total_mass, mpdata.total_mass, decimal=5)

	corr = np.corrcoef(py_pde.result.ravel(), mpdata.result.ravel())[0, 1]
	assert_almost_equal(corr, 1,  decimal=5)

	diff = py_pde.result - mpdata.result
	rmse = np.sqrt(np.mean(diff ** 2))
	assert_almost_equal(rmse, 0, decimal=5)

	l1_error = np.sum(np.abs(diff)) * dx * dy
	assert_almost_equal(l1_error, 0, decimal=5)

	l2_norm = np.linalg.norm(solution_pymp - solution_pypde)
	assert l2_norm < 1e-2

	expected: Two2DiffusionSolution = analytical_solution()
	assert py_pde.result.shape == expected.result.shape

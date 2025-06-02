import numpy as np
import pytest
from numpy.ma.testutils import assert_almost_equal

from examples.PyMPDATA_examples.comparison_against_pypde_et_al_2025.diffusion_2d import (
	py_pde_solution,
	mpdata_solution,
	InitialConditions,
	Two2DiffusionSolution,
)


@pytest.fixture()
def initial_conditions() -> InitialConditions:
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
	py_pde_result: Two2DiffusionSolution = py_pde_solution(
		initial_conditions=initial_conditions,
	)
	mpdata_result: Two2DiffusionSolution = mpdata_solution(
		initial_conditions=initial_conditions,
	)

	# sanitiy checks
	assert (py_pde_result.initial_conditions == mpdata_result.initial_conditions).all()

	assert py_pde_result.shape == mpdata_result.shape

	assert np.all(np.isfinite(py_pde_result))
	assert np.all(np.isfinite(mpdata_result))

	assert np.all(py_pde_result >= 0)
	assert np.all(mpdata_result >= 0)

	# compare results
	assert np.allclose(py_pde_result, mpdata_result, rtol=1e-6, atol=1e-8)

	# total mass check
	assert_almost_equal(py_pde_result.sum(), mpdata_result.sum(), decimal=5)

	corr = np.corrcoef(py_pde_result.ravel(), mpdata_result.ravel())[0, 1]
	assert_almost_equal(corr, 1, decimal=5)

	diff = py_pde_result - mpdata_result

	rmse = np.sqrt(np.mean(diff ** 2))
	assert_almost_equal(rmse, 0, decimal=5)

	l1_error = np.sum(np.abs(diff)) * initial_conditions.dx * initial_conditions.dy
	assert_almost_equal(l1_error, 0, decimal=5)

	l2_norm = np.linalg.norm(py_pde_result - mpdata_result)
	assert_almost_equal(l2_norm, 0, decimal=5)

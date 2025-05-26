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
		time_step_size=0.001,
		time_end=2,
		shape=(30, 18),
		range_x=(-1.0, 1.0),
		range_y=(0.0, 2.0),
	)


def test_similarity_of_solutions(initial_conditions: InitialConditions) -> None:
	seed = 2137

	py_pde: Two2DiffusionSolution = py_pde_solution(
		initial_conditions=initial_conditions,
		random=np.random.RandomState(seed=seed),
	)
	py_pde2: Two2DiffusionSolution = py_pde_solution(
		initial_conditions=initial_conditions,
		random=np.random.RandomState(seed=seed),
	)

	mpdata: Two2DiffusionSolution = mpdata_solution(
		initial_conditions=initial_conditions,
		random=np.random.RandomState(seed=seed),
	)
	mpdata2: Two2DiffusionSolution = mpdata_solution(
		initial_conditions=initial_conditions,
		random=np.random.RandomState(seed=seed),
	)

	# sanitiy checks
	assert np.allclose(py_pde.initial_conditions, mpdata.initial_conditions)

	assert np.allclose(py_pde, py_pde2)
	assert np.allclose(mpdata, mpdata2)

	assert py_pde.result.shape == mpdata.result.shape

	assert np.all(np.isfinite(py_pde.result))
	assert np.all(np.isfinite(mpdata.result))

	assert np.all(py_pde.result >= 0)
	assert np.all(mpdata.result >= 0)

	# compare results
	assert np.allclose(py_pde.result, mpdata.result, rtol=1e-6, atol=1e-8)

	assert_almost_equal(py_pde.total_mass, mpdata.total_mass, decimal=5)

	corr = np.corrcoef(py_pde.result.ravel(), mpdata.result.ravel())[0, 1]
	assert_almost_equal(corr, 1, decimal=5)

	diff = py_pde.result - mpdata.result

	rmse = np.sqrt(np.mean(diff ** 2))
	assert_almost_equal(rmse, 0, decimal=5)

	l1_error = np.sum(np.abs(diff)) * initial_conditions.dx * initial_conditions.dy
	assert_almost_equal(l1_error, 0, decimal=5)

	l2_norm = np.linalg.norm(py_pde.result - mpdata.result)
	assert_almost_equal(l2_norm, 0, decimal=5)

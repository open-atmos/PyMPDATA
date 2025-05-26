from examples.PyMPDATA_examples.comparison_against_pypde_et_al_2025.diffusion_2d import py_pde_solution, mpdata_solution


def test_similarity_of_solutions():
	py_pde = py_pde_solution()
	mpdata = mpdata_solution()


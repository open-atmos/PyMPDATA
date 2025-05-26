from pde import CartesianGrid, DiffusionPDE, ScalarField

def py_pde_solution():
	grid = CartesianGrid([[-1, 1], [0, 2]], [30, 18])  # generate grid
	state = ScalarField(grid)  # generate initial condition
	state.insert([0, 1], 1)
	state.plot(title="Initial condition", cmap="magma")
	eq = DiffusionPDE(0.1)  # define the pde
	result_pdepde = eq.solve(state, t_range=1, dt=0.001)
	result_pdepde.plot(cmap="magma")
	result_pdepde = result_pdepde.data
	return result_pdepde

import numpy as np
import matplotlib.pyplot as plt

from PyMPDATA import ScalarField, VectorField, Stepper, Solver, Options
from PyMPDATA.boundary_conditions import Periodic

def mpdata_solution():
	D = 0.1  # Diffusion coefficient
	dt = 0.001  # Time step size
	t_end = 2
	nx, ny = 30, 18
	min_x, max_x = -1.0, 1.0
	min_y, max_y = 0.0, 2.0

	dx = (max_x - min_x) / nx
	dy = (max_y - min_y) / ny
	print(dx, dy)
	n_steps = int(t_end / dt)  # Number of time steps

	# ------------------------
	# Grid: physical coordinates
	# ------------------------
	x = np.linspace(min_x + dx / 2, max_x - dx / 2, nx)
	y = np.linspace(min_y + dy / 2, max_y - dy / 2, ny)

	# ------------------------
	# Gaussian blob initializer
	# ------------------------
	def init_py_pde_like_pulse():
		data = np.zeros((nx, ny))

		# Locate cell nearest (0, 1)
		i = np.argmin(np.abs(x - 0.0))
		j = np.argmin(np.abs(y - 1.0))

		# Distribute mass over 2x2 cells (py-pde seems to do this internally)
		mass_per_cell = 1.0 / (4 * dx * dy)
		data[i, j] = mass_per_cell
		data[i + 1, j] = mass_per_cell
		data[i, j + 1] = mass_per_cell
		data[i + 1, j + 1] = mass_per_cell

		return data

	# ------------------------
	# Options and stepper
	# ------------------------
	opt = Options(n_iters=2, non_zero_mu_coeff=True)
	stepper = Stepper(options=opt, n_dims=2)
	halo = opt.n_halo  # Get required halo size

	# ------------------------
	# Initialize scalar field
	# ------------------------
	data = init_py_pde_like_pulse()
	data_mpdata = data
	advectee = ScalarField(data=data, halo=halo, boundary_conditions=(Periodic(), Periodic()))

	# ------------------------
	# No advection, so velocities are 0
	# ------------------------
	Cx = np.zeros((nx + 1, ny), dtype=opt.dtype)
	Cy = np.zeros((nx, ny + 1), dtype=opt.dtype)
	advector = VectorField(data=(Cx, Cy), halo=halo, boundary_conditions=(Periodic(), Periodic()))

	# ------------------------
	# Create solver
	# ------------------------
	solver = Solver(stepper=stepper, advector=advector, advectee=advectee)

	# ------------------------
	# Run the simulation
	# ------------------------
	mu_x = (D * dt / dx ** 2) / 2
	mu_y = (D * dt / dy ** 2) / 2
	print(f"mu_x={mu_x:.3f}, mu_y={mu_y:.3f}")
	solver.advance(n_steps=n_steps, mu_coeff=(mu_x, mu_y))
	result_mpdata = solver.advectee.get()
	return result_mpdata

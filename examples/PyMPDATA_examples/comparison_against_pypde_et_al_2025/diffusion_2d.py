from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


class InitialConditions:
	def __init__(
		self,
		diffusion_coefficient: float,
		time_step_size: float,
		time_end: float,
		shape: tuple[int, int],
		range_x: tuple[float, float],
		range_y: tuple[float, float],
	) -> None:
		self.diffusion_coefficient = diffusion_coefficient
		self.time_step_size = time_step_size
		self.time_end = time_end
		self.nx, self.ny = shape
		self.min_x, self.max_x = range_x
		self.min_y, self.max_y = range_y

	@property
	def dx(self) -> float:
		return (self.max_x - self.min_x) / self.nx

	@property
	def dy(self) -> float:
		return (self.max_y - self.min_y) / self.ny

	@property
	def n_steps(self) -> int:
		return int(self.time_end / self.time_step_size)

	def create_data(self) -> npt.NDArray[np.float64]:
		x = np.linspace(self.min_x + self.dx / 2, self.max_x - self.dx / 2, self.nx)
		y = np.linspace(self.min_y + self.dy / 2, self.max_y - self.dy / 2, self.ny)
		data = np.zeros((self.nx, self.ny))
		# Locate cell nearest (0, 1)
		i = np.argmin(np.abs(x - 0.0))
		j = np.argmin(np.abs(y - 1.0))
		# Distribute mass over 2x2 cells (py-pde seems to do this internally)
		mass_per_cell = 1.0 / (4 * self.dx * self.dy)
		data[i, j] = mass_per_cell
		data[i + 1, j] = mass_per_cell
		data[i, j + 1] = mass_per_cell
		data[i + 1, j + 1] = mass_per_cell
		return data


@dataclass(frozen=True)
class Two2DiffusionSolution:
	result: npt.NDArray[np.float64]
	initial_conditions: InitialConditions

	@property
	def total_mass(self) -> float:
		return self.result.sum()  # * dx * dy


def py_pde_solution(
	initial_conditions: InitialConditions,
	random: np.random.RandomState,
) -> Two2DiffusionSolution:
	"""To be moved from notebook."""


def mpdata_solution(
	initial_conditions: InitialConditions,
	random: np.random.RandomState,
) -> Two2DiffusionSolution:
	"""To be moved from notebook."""

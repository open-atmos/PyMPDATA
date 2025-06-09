"""
Test the similarity of solutions from MPDATA and PyPDE for 2D diffusion
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pde import CartesianGrid, DiffusionPDE
from pde import ScalarField as PDEScalarField

from PyMPDATA import Options, ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Periodic


@dataclass
class InitialConditions:
    """
    Initial conditions for the 2D diffusion problem.
    """

    def __init__(
        self,
        *,
        diffusion_coefficient: float,
        time_step: float,
        time_end: float,
        grid_shape: tuple[int, int],
        grid_range_x: tuple[float, float],
        grid_range_y: tuple[float, float],
        pulse_position: tuple[float, float],
    ) -> None:
        self.diffusion_coefficient = diffusion_coefficient
        self.time_step = time_step
        self.time_end = time_end
        self.grid_shape = grid_shape
        self.grid_range_x = grid_range_x
        self.grid_range_y = grid_range_y
        self.pulse_position = pulse_position
        self.nx, self.ny = grid_shape
        self.min_x, self.max_x = grid_range_x
        self.min_y, self.max_y = grid_range_y
        self.pulse_x, self.pulse_y = pulse_position

    def __repr__(self) -> str:
        return (
            f"InitialConditions(diffusion_coefficient={self.diffusion_coefficient}, "
            f"time_step={self.time_step}, time_end={self.time_end}, "
            f"grid_shape={self.grid_shape}, grid_range_x={self.grid_range_x}, "
            f"grid_range_y={self.grid_range_y}, pulse_position={self.pulse_position})"
        )

    @property
    def dx(self) -> float:
        """Calculate the grid spacing in the x-direction."""
        return (self.max_x - self.min_x) / self.nx

    @property
    def dy(self) -> float:
        """Calculate the grid spacing in the y-direction."""
        return (self.max_y - self.min_y) / self.ny

    @property
    def n_steps(self) -> int:
        """Calculate the number of time steps based on the time range and time step."""
        return int(self.time_end / self.time_step)


type Two2DiffusionSolution = npt.NDArray[np.float64]


def py_pde_solution(initial_conditions: InitialConditions) -> Two2DiffusionSolution:
    """
    Solve the 2D diffusion equation using PyPDE.
    """
    grid = CartesianGrid(
        bounds=[
            initial_conditions.grid_range_x,
            initial_conditions.grid_range_y,
        ],
        shape=initial_conditions.grid_shape,
    )
    state = PDEScalarField(grid=grid)
    state.insert(
        point=np.array(initial_conditions.pulse_position),
        amount=1,
    )
    eq = DiffusionPDE(diffusivity=initial_conditions.diffusion_coefficient)
    result = eq.solve(
        state=state,
        t_range=1,
        dt=initial_conditions.time_step,
    )
    return result.data


def mpdata_solution(initial_conditions: InitialConditions) -> Two2DiffusionSolution:
    """
    Solve the 2D diffusion equation using PyMPDATA.
    """
    opt = Options(
        n_iters=2,
        non_zero_mu_coeff=True,
    )
    stepper = Stepper(
        options=opt,
        n_dims=2,
    )

    def create_pde_like_data(ic) -> npt.NDArray[np.float64]:
        """
        Create a 2D array with a pulse at the specified position.
        """
        x = np.linspace(ic.min_x + ic.dx / 2, ic.max_x - ic.dx / 2, ic.nx)
        y = np.linspace(ic.min_y + ic.dy / 2, ic.max_y - ic.dy / 2, ic.ny)
        result = np.zeros((ic.nx, ic.ny))
        # Locate cell nearest (0, 1)
        i = np.argmin(np.abs(x - ic.pulse_x))
        j = np.argmin(np.abs(y - ic.pulse_y))
        # Distribute mass over 2x2 cells (py-pde seems to do this internally)
        mass_per_cell = 1.0 / (4 * ic.dx * ic.dy)
        result[i, j] = mass_per_cell
        result[i + 1, j] = mass_per_cell
        result[i, j + 1] = mass_per_cell
        result[i + 1, j + 1] = mass_per_cell
        return result

    data = create_pde_like_data(ic=initial_conditions)

    advectee = ScalarField(
        data=data, halo=opt.n_halo, boundary_conditions=(Periodic(), Periodic())
    )

    cx = np.zeros(
        shape=(initial_conditions.nx + 1, initial_conditions.ny),
        dtype=opt.dtype,
    )
    cy = np.zeros(
        shape=(initial_conditions.nx, initial_conditions.ny + 1),
        dtype=opt.dtype,
    )
    advector = VectorField(
        data=(cx, cy),
        halo=opt.n_halo,
        boundary_conditions=(Periodic(), Periodic()),
    )

    solver = Solver(
        stepper=stepper,
        advector=advector,
        advectee=advectee,
    )
    mu_x = (
        initial_conditions.diffusion_coefficient
        * initial_conditions.time_step
        / initial_conditions.dx**2
    )
    mu_y = (
        initial_conditions.diffusion_coefficient
        * initial_conditions.time_step
        / initial_conditions.dy**2
    )
    solver.advance(
        n_steps=initial_conditions.n_steps,
        mu_coeff=(mu_x, mu_y),
    )
    return solver.advectee.get()

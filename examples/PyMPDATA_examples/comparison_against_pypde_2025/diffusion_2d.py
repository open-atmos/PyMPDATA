"""
Test the similarity of solutions from MPDATA and PyPDE for 2D diffusion
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pde import (
    CartesianGrid,
    DataFieldBase,
    DiffusionPDE,
)
from pde import ScalarField as PDEScalarField

from PyMPDATA import Options, ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Periodic


@dataclass
class InitialConditions:
    """
    Initial conditions for the 2D diffusion problem.
    """

    N_DIMS = 2

    diffusion_coefficient: float
    time_step: float
    time_end: float
    grid_shape: tuple[int, int]
    grid_range_x: tuple[float, float]
    grid_range_y: tuple[float, float]
    pulse_position: tuple[float, float]
    pulse_shape: tuple[int, int]

    @property
    def nx(self) -> int:
        return self.grid_shape[0]

    @property
    def ny(self) -> int:
        return self.grid_shape[1]

    @property
    def min_x(self) -> float:
        return self.grid_range_x[0]

    @property
    def max_x(self) -> float:
        return self.grid_range_x[1]

    @property
    def min_y(self) -> float:
        return self.grid_range_y[0]

    @property
    def max_y(self) -> float:
        return self.grid_range_y[1]

    @property
    def pulse_x(self) -> float:
        return self.pulse_position[0]

    @property
    def pulse_y(self) -> float:
        return self.pulse_position[1]

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


def create_pde_state(initial_conditions: InitialConditions) -> DataFieldBase:
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
    return state


Grid = npt.NDArray[np.float64]


def py_pde_solution(initial_conditions: InitialConditions) -> Grid:
    """
    Solve the 2D diffusion equation using PyPDE.
    """
    state = create_pde_state(initial_conditions=initial_conditions)
    eq = DiffusionPDE(diffusivity=initial_conditions.diffusion_coefficient)
    result = eq.solve(
        state=state,
        t_range=1,
        dt=initial_conditions.time_step,
    )
    if result is not None and hasattr(result, "data"):
        return result.data
    raise RuntimeError(
        "PyPDE solve did not return a valid result with a 'data' attribute."
    )


def mpdata_solution(initial_conditions: InitialConditions) -> Grid:
    """
    Solve the 2D diffusion equation using PyMPDATA.
    """
    opt = Options(
        n_iters=2,
        non_zero_mu_coeff=True,
    )
    stepper = Stepper(
        options=opt,
        n_dims=initial_conditions.N_DIMS,
    )

    data = create_pde_state(initial_conditions=initial_conditions).data

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

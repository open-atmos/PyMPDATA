# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring,invalid-name,too-few-public-methods,too-many-locals

import numpy as np
from PyMPDATA import ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Periodic

from .domain_decomposition import mpi_indices
from .periodic import MPIPeriodic


class Simulation:
    def __init__(
        self,
        *,
        mpdata_options,
        n_threads,
        grid,
        rank,
        size,
        initial_condition,
        courant_field,
    ):
        halo = mpdata_options.n_halo

        xi, yi = mpi_indices(grid, rank, size)
        nx, ny = xi.shape

        boundary_conditions = (MPIPeriodic(size=size), Periodic())
        self.advectee = ScalarField(
            data=initial_condition(xi, yi, grid),
            halo=halo,
            boundary_conditions=boundary_conditions,
        )

        advector = VectorField(
            data=(
                np.full((nx + 1, ny), courant_field[0]),
                np.full((nx, ny + 1), courant_field[1]),
            ),
            halo=halo,
            boundary_conditions=boundary_conditions,
        )
        stepper = Stepper(
            options=mpdata_options,
            n_dims=2,
            n_threads=n_threads,
            left_first=rank % 2 == 0,
        )
        self.solver = Solver(stepper=stepper, advectee=self.advectee, advector=advector)

    def advance(self, n_steps):
        self.solver.advance(n_steps)

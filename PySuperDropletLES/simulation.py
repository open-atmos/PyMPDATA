# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring,invalid-name,too-many-locals

import numpy as np
from PyMPDATA import Options, ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Periodic
from PyMPDATA.impl.domain_decomposition import make_subdomain


class Simulation:
    subdomain = make_subdomain(jit_flags={})

    def __init__(
        self, *, n_iters, n_threads, grid, rank, size, initial_condition, courant_field
    ):
        options = Options(n_iters=n_iters)
        halo = options.n_halo

        xi, yi = Simulation.mpi_indices(grid, rank, size)
        nx, ny = xi.shape

        boundary_conditions = (Periodic(), Periodic())
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
        stepper = Stepper(options=options, n_dims=2, n_threads=n_threads)
        self.solver = Solver(stepper=stepper, advectee=self.advectee, advector=advector)

    def advance(self, n_steps):
        self.solver.advance(n_steps)

    @staticmethod
    def mpi_indices(grid, rank, size):
        start, stop = Simulation.subdomain(grid[0], rank, size)
        xi, yi = np.indices((stop - start, grid[1]), dtype=float)
        xi += start
        return xi, yi

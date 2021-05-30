import numpy as np

from .stepper import Stepper
from .arakawa_c.vector_field import VectorField
from .arakawa_c.scalar_field import ScalarField
from .solver import Solver
from .options import Options
from .arakawa_c.discretisation import nondivergent_vector_field_2d
from .arakawa_c.boundary_condition.periodic_boundary_condition import PeriodicBoundaryCondition


class Factories:
    @staticmethod
    def constant_1d(data: np.ndarray, C: float, options: Options):
        solver = Solver(
            stepper=Stepper(options=options, n_dims=len(data.shape), non_unit_g_factor=False),
            advectee=ScalarField(data.astype(options.dtype), halo=options.n_halo, boundary_conditions=(PeriodicBoundaryCondition(),)),
            advector=VectorField((np.full(data.shape[0] + 1, C, dtype=options.dtype),), halo=options.n_halo, boundary_conditions=(PeriodicBoundaryCondition(),))
        )
        return solver

    @staticmethod
    def constant_2d(data: np.ndarray, C, options: Options, grid_static=True):
        grid = data.shape
        advector_data = [
            np.full((grid[0] + 1, grid[1]), C[0], dtype=options.dtype),
            np.full((grid[0], grid[1] + 1), C[1], dtype=options.dtype)
        ]
        advector = VectorField(advector_data, halo=options.n_halo, boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition()))
        advectee = ScalarField(data=data.astype(dtype=options.dtype), halo=options.n_halo, boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition()))
        if grid_static:
            stepper = Stepper(options=options, grid=grid, non_unit_g_factor=False)
        else:
            stepper = Stepper(options=options, n_dims=2, non_unit_g_factor=False)
        mpdata = Solver(stepper=stepper, advectee=advectee, advector=advector)
        return mpdata

    @staticmethod
    def stream_function_2d_basic(grid, size, dt, stream_function, field: np.ndarray, options: Options):
        stepper = Stepper(options=options, grid=grid, non_unit_g_factor=False)
        advector = nondivergent_vector_field_2d(grid, size, dt, stream_function, options.n_halo)
        advectee = ScalarField(field.astype(dtype=options.dtype), halo=options.n_halo, boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition()))
        return Solver(stepper=stepper, advectee=advectee, advector=advector)

    @staticmethod
    def stream_function_2d(grid, size, dt, stream_function, field_values, g_factor: np.ndarray, options: Options):
        stepper = Stepper(options=options, grid=grid, non_unit_g_factor=True)
        advector = nondivergent_vector_field_2d(grid, size, dt, stream_function, options.n_halo)
        g_factor = ScalarField(g_factor.astype(dtype=options.dtype), halo=options.n_halo,
                               boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition()))
        mpdatas = {}
        for k, v in field_values.items():
            advectee = ScalarField(np.full(grid, v, dtype=options.dtype), halo=options.n_halo,
                                   boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition()))
            mpdatas[k] = Solver(stepper=stepper, advectee=advectee, advector=advector, g_factor=g_factor)
        return advector, mpdatas

    @staticmethod
    def advection_diffusion_1d(*,
                               options: Options,
                               advectee: np.ndarray,
                               advector: float,
                               boundary_conditions
                               ):
        assert advectee.ndim == 1
        grid = advectee.shape
        stepper = Stepper(options=options, n_dims=len(grid), non_unit_g_factor=False)
        return Solver(stepper=stepper,
                      advectee=ScalarField(advectee.astype(dtype=options.dtype), halo=options.n_halo, boundary_conditions=boundary_conditions),
                      advector=VectorField((np.full(grid[0]+1, advector, dtype=options.dtype),), halo=options.n_halo, boundary_conditions=boundary_conditions)
                      )


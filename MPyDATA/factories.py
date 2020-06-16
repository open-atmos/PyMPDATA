"""
Created at 21.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np

from .arakawa_c.vector_field import VectorField
from .arakawa_c.scalar_field import ScalarField
from .solver import Solver
from .options import Options
from MPyDATA.stepper import Stepper
from .arakawa_c.discretisation import nondivergent_vector_field_2d, discretised_analytical_solution
from .arakawa_c.boundary_condition.extrapolated_boundary_condition import ExtrapolatedBoundaryCondition
from .arakawa_c.boundary_condition.constant_boundary_condition import ConstantBoundaryCondition
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
        GC_data = [
            np.full((grid[0] + 1, grid[1]), C[0], dtype=options.dtype),
            np.full((grid[0], grid[1] + 1), C[1], dtype=options.dtype)
        ]
        GC = VectorField(GC_data, halo=options.n_halo, boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition()))
        state = ScalarField(data=data.astype(dtype=options.dtype), halo=options.n_halo, boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition()))
        if grid_static:
            stepper = Stepper(options=options, grid=grid, non_unit_g_factor=False)
        else:
            stepper = Stepper(options=options, n_dims=2, non_unit_g_factor=False)
        mpdata = Solver(stepper=stepper, advectee=state, advector=GC)
        return mpdata

    @staticmethod
    def stream_function_2d_basic(grid, size, dt, stream_function, field: np.ndarray, options: Options):
        step = Stepper(options=options, grid=grid, non_unit_g_factor=False)
        GC = nondivergent_vector_field_2d(grid, size, dt, stream_function, options.n_halo)
        advectee = ScalarField(field.astype(dtype=options.dtype), halo=options.n_halo, boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition()))
        return Solver(stepper=step, advectee=advectee, advector=GC)

    @staticmethod
    def stream_function_2d(grid, size, dt, stream_function, field_values, g_factor: np.ndarray, options: Options):
        step = Stepper(options=options, grid=grid, non_unit_g_factor=True)
        GC = nondivergent_vector_field_2d(grid, size, dt, stream_function, options.n_halo)
        g_factor = ScalarField(g_factor.astype(dtype=options.dtype), halo=options.n_halo, boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition()))
        mpdatas = {}
        for k, v in field_values.items():
            advectee = ScalarField(np.full(grid, v, dtype=options.dtype), halo=options.n_halo, boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition()))
            mpdatas[k] = Solver(stepper=step, advectee=advectee, advector=GC, g_factor=g_factor)
        return GC, mpdatas

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

    # TODO: move to Olesik_et_al_2020 example?
    @staticmethod
    def condensational_growth(nr, r_min, r_max, GC_max, grid_layout, psi_coord, pdf_of_r, drdt_of_r, opts: Options):
        # psi = psi(p)
        dp_dr = psi_coord.dx_dr
        dx_dr = grid_layout.dx_dr

        xh, dx = np.linspace(
            grid_layout.x(r_min),
            grid_layout.x(r_max),
            nr + 1,
            retstep=True
        )
        rh = grid_layout.r(xh)

        x = np.linspace(
            xh[0] + dx / 2,
            xh[-1] - dx / 2,
            nr
        )
        r = grid_layout.r(x)

        def pdf_of_r_over_psi(r):
            return pdf_of_r(r)/ psi_coord.dx_dr(r)

        psi = discretised_analytical_solution(rh, pdf_of_r_over_psi)

        dp_dt = drdt_of_r(rh) * dp_dr(rh)
        G = dp_dr(r) / dx_dr(r)

        # C = dr_dt * dt / dr
        # GC = dp_dr / dx_dr * dr_dt * dt / dr =
        #        \       \_____ / _..____/
        #         \_____.._____/    \_ dt/dx
        #               |
        #             dp_dt

        dt = GC_max * dx / np.amax(dp_dt)
        GCh = dp_dt * dt / dx

        # CFL condition
        np.testing.assert_array_less(np.abs(GCh), 1)

        g_factor = ScalarField(G.astype(dtype=opts.dtype), halo=opts.n_halo, boundary_conditions=(ExtrapolatedBoundaryCondition(),))
        state = ScalarField(psi.astype(dtype=opts.dtype), halo=opts.n_halo, boundary_conditions=(ConstantBoundaryCondition(0),))
        GC_field = VectorField([GCh.astype(dtype=opts.dtype)], halo=opts.n_halo, boundary_conditions=(ConstantBoundaryCondition(0),))
        stepper = Stepper(
            options=opts,
            n_dims=1,
            non_unit_g_factor=True
        )
        return (
            Solver(stepper=stepper, g_factor=g_factor, advectee=state, advector=GC_field),
            r,
            rh,
            dx,
            dt
        )





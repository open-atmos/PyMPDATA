import math

import numpy as np
from PyMPDATA_examples.utils.discretisation import discretised_analytical_solution

from PyMPDATA import Options, ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Constant, Extrapolated


class Simulation:
    @staticmethod
    def make_condensational_growth_solver(
        nr,
        r_min,
        r_max,
        GC_max,
        grid_layout,
        psi_coord,
        pdf_of_r,
        drdt_of_r,
        opts: Options,
    ):
        # psi = psi(p)
        dp_dr = psi_coord.dx_dr
        dx_dr = grid_layout.dx_dr

        xh, dx = np.linspace(
            grid_layout.x(r_min), grid_layout.x(r_max), nr + 1, retstep=True
        )
        rh = grid_layout.r(xh)

        x = np.linspace(xh[0] + dx / 2, xh[-1] - dx / 2, nr)
        r = grid_layout.r(x)

        def pdf_of_r_over_psi(r):
            return pdf_of_r(r) / psi_coord.dx_dr(r)

        psi = discretised_analytical_solution(
            rh, pdf_of_r_over_psi, midpoint_value=True, r=r
        )

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

        g_factor = ScalarField(
            G.astype(dtype=opts.dtype),
            halo=opts.n_halo,
            boundary_conditions=(Extrapolated(),),
        )
        state = ScalarField(
            psi.astype(dtype=opts.dtype),
            halo=opts.n_halo,
            boundary_conditions=(Constant(0),),
        )
        GC_field = VectorField(
            [GCh.astype(dtype=opts.dtype)],
            halo=opts.n_halo,
            boundary_conditions=(Constant(0),),
        )
        stepper = Stepper(options=opts, n_dims=1, non_unit_g_factor=True)
        return (
            Solver(
                stepper=stepper, g_factor=g_factor, advectee=state, advector=GC_field
            ),
            r,
            rh,
            dx,
            dt,
            g_factor,
        )

    @staticmethod
    def __mgn(quantity, unit):
        return quantity.to(unit).magnitude

    def __init__(self, settings, grid_layout, psi_coord, opts, GC_max):
        self.settings = settings
        self.psi_coord = psi_coord
        self.grid_layout = grid_layout

        # units of calculation
        self.__t_unit = self.settings.si.seconds
        self.__r_unit = self.settings.si.micrometre
        self.__p_unit = psi_coord.x(self.__r_unit)
        self.__n_of_r_unit = (
            self.settings.si.centimetres**-3 / self.settings.si.micrometre
        )

        (
            self.solver,
            self.__r,
            self.__rh,
            self.dx,
            dt,
            self._g_factor,
        ) = Simulation.make_condensational_growth_solver(
            self.settings.nr,
            self.__mgn(self.settings.r_min, self.__r_unit),
            self.__mgn(self.settings.r_max, self.__r_unit),
            GC_max,
            grid_layout,
            psi_coord,
            lambda r: self.__mgn(
                self.settings.pdf(r * self.__r_unit), self.__n_of_r_unit
            ),
            lambda r: self.__mgn(
                self.settings.drdt(r * self.__r_unit), self.__r_unit / self.__t_unit
            ),
            opts,
        )

        self.out_steps = tuple(math.ceil(t / dt) for t in settings.out_times)
        self.dt = dt * self.__t_unit

    def step(self, nt):
        return self.solver.advance(nt)

    @property
    def r(self):
        return self.__r * self.__r_unit

    @property
    def rh(self):
        return self.__rh * self.__r_unit

    @property
    def n_of_r(self):
        psi = self.solver.advectee.get()
        n = psi * self.psi_coord.dx_dr(self.__r)
        return n * self.__n_of_r_unit

    @property
    def psi(self):
        psi_unit = self.__n_of_r_unit * self.__r_unit / self.__p_unit
        return self.solver.advectee.get() * psi_unit

    @property
    def g_factor(self):
        return self._g_factor.get()

    @property
    def dp_dr(self):
        return self.psi_coord.dx_dr(self.__r)

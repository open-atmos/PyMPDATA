from functools import cached_property

import numpy as np
from PyMPDATA_examples.utils.discretisation import discretised_analytical_solution

from PyMPDATA import Options, ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Extrapolated
from PyMPDATA.impl.enumerations import INNER, OUTER


# pylint: disable=too-few-public-methods
class Settings:
    def __init__(self, T, sgma, r, K, S_min, S_max):
        self.T = T
        self.sgma = sgma
        self.r = r
        self.K = K
        self.S_min = S_min
        self.S_max = S_max
        self.rh = None

    def payoff(self, A: np.ndarray, da: np.float32 = 0, variant="call"):
        def call(x):
            return np.maximum(0, x - self.K)

        def put(x):
            return np.maximum(0, self.K - x)

        if variant == "call":
            payoff_func = call
        else:
            payoff_func = put

        self.rh = np.linspace(A[0] - da / 2, A[-1] + da / 2, len(A) + 1)
        output = discretised_analytical_solution(self.rh, payoff_func)
        return output


class Simulation:
    def __init__(self, settings, *, nx, ny, nt, OPTIONS, variant="call"):
        self.nx = nx
        self.nt = nt
        self.settings = settings
        self.ny = ny
        self.dt = settings.T / self.nt
        log_s_min = np.log(settings.S_min)
        log_s_max = np.log(settings.S_max)
        self.S = np.exp(np.linspace(log_s_min, log_s_max, self.nx))
        self.A, self.dy = np.linspace(
            0, settings.S_max, self.ny, retstep=True, endpoint=True
        )
        self.dx = (log_s_max - log_s_min) / self.nx
        self.settings = settings
        sigma_squared = pow(settings.sgma, 2)
        courant_number_x = -(0.5 * sigma_squared - settings.r) * (-self.dt) / self.dx
        self.l2 = self.dx * self.dx / sigma_squared / self.dt
        self.mu_coeff = (0.5 / self.l2, 0)
        assert (
            self.l2 > 2
        ), f"Lambda squared should be more than 2 for stability {self.l2}"
        self.payoff = settings.payoff(A=self.A, da=self.dy, variant=variant)
        options = Options(**OPTIONS)
        stepper = Stepper(options=options, n_dims=2)
        x_dim_advector = np.full(
            (self.nx + 1, self.ny),
            courant_number_x,
            dtype=options.dtype,
        )
        cfl_condition = np.max(np.abs(self.a_dim_advector)) + np.max(
            np.abs(x_dim_advector)
        )
        assert cfl_condition < 1, f"CFL condition not met {cfl_condition}"
        self.solver = Solver(
            stepper=stepper,
            advectee=ScalarField(
                self.payoff_2d.astype(dtype=options.dtype)
                * np.exp(-self.settings.r * self.settings.T),
                halo=options.n_halo,
                boundary_conditions=self.boundary_conditions,
            ),
            advector=VectorField(
                (x_dim_advector, self.a_dim_advector),
                halo=options.n_halo,
                boundary_conditions=self.boundary_conditions,
            ),
        )
        self.rhs = np.zeros((self.nx, self.ny))

    @property
    def payoff_2d(self):
        raise NotImplementedError()

    @property
    def a_dim_advector(self):
        raise NotImplementedError()

    @property
    def boundary_conditions(self):
        return (
            Extrapolated(OUTER),
            Extrapolated(INNER),
        )

    def step(self, nt=1):
        self.solver.advance(n_steps=nt, mu_coeff=self.mu_coeff)


class AsianArithmetic(Simulation):
    @cached_property
    def payoff_2d(self):
        return np.repeat([self.payoff], self.nx, axis=0)

    @property
    def a_dim_advector(self):
        a_dim_advector = np.zeros((self.nx, self.ny + 1))
        for i in range(self.ny + 1):
            a_dim_advector[:, i] = -self.dt / self.dy * self.S / self.settings.T
        return a_dim_advector

import numba
import numpy as np
from PyMPDATA_examples.Arabas_and_Farhat_2020.options import OPTIONS

from PyMPDATA import Options, ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Extrapolated
from PyMPDATA.impl.enumerations import ARG_DATA


class Simulation:
    @staticmethod
    def _factory(
        *, options: Options, advectee: np.ndarray, advector: float, boundary_conditions
    ):
        stepper = Stepper(
            options=options, n_dims=len(advectee.shape), non_unit_g_factor=False
        )
        return Solver(
            stepper=stepper,
            advectee=ScalarField(
                advectee.astype(dtype=options.dtype),
                halo=options.n_halo,
                boundary_conditions=boundary_conditions,
            ),
            advector=VectorField(
                (np.full(advectee.shape[0] + 1, advector, dtype=options.dtype),),
                halo=options.n_halo,
                boundary_conditions=boundary_conditions,
            ),
        )

    def __init__(self, settings):
        self.settings = settings

        sigma2 = pow(settings.sigma, 2)
        dx_opt = abs(
            settings.C_opt / (0.5 * sigma2 - settings.r) * settings.l2_opt * sigma2
        )
        dt_opt = pow(dx_opt, 2) / sigma2 / settings.l2_opt

        # adjusting dt so that nt is integer
        self.dt = settings.T
        self.nt = 0
        while self.dt > dt_opt:
            self.nt += 1
            self.dt = settings.T / self.nt

        # adjusting dx to match requested l^2
        dx = np.sqrt(settings.l2_opt * self.dt) * settings.sigma

        # calculating actual u number and lambda
        self.C = -(0.5 * sigma2 - settings.r) * (-self.dt) / dx
        self.l2 = dx * dx / sigma2 / self.dt

        # adjusting nx and setting S_beg, S_end
        S_beg = settings.S_match
        self.nx = 1
        while S_beg > settings.S_min:
            self.nx += 1
            S_beg = np.exp(np.log(settings.S_match) - self.nx * dx)

        self.ix_match = self.nx

        S_end = settings.S_match
        while S_end < settings.S_max:
            self.nx += 1
            S_end = np.exp(np.log(S_beg) + (self.nx - 1) * dx)

        # asset price
        self.S = np.exp(np.log(S_beg) + np.arange(self.nx) * dx)

        self.mu_coeff = (0.5 / self.l2,)
        self.solvers = {}
        self.solvers[1] = self._factory(
            advectee=settings.payoff(self.S),
            advector=self.C,
            options=Options(n_iters=1, non_zero_mu_coeff=True),
            boundary_conditions=(Extrapolated(),),
        )
        self.solvers[2] = self._factory(
            advectee=settings.payoff(self.S),
            advector=self.C,
            options=Options(**OPTIONS),
            boundary_conditions=(Extrapolated(),),
        )

    def run(self, n_iters: int):
        if self.settings.amer:
            psi = self.solvers[n_iters].advectee.data
            f_T = np.empty_like(psi)
            f_T[:] = psi[:] / np.exp(-self.settings.r * self.settings.T)
            T = self.settings.T
            r = self.settings.r
            dt = self.dt

            @numba.experimental.jitclass([])
            class PostStep:
                # pylint: disable=too-few-public-methods
                def __init__(self):
                    pass

                def call(self, _, psi, step, index):
                    t = T - (step + 1) * dt
                    psi[index].field[ARG_DATA][:] += (
                        np.maximum(psi[index].field[ARG_DATA], f_T / np.exp(r * t))
                        - psi[index].field[ARG_DATA]
                    )

            self.solvers[n_iters].advance(
                self.nt, mu_coeff=self.mu_coeff, post_step=PostStep()
            )
        else:
            self.solvers[n_iters].advance(self.nt, mu_coeff=self.mu_coeff)

        return self.solvers[n_iters].advectee.get()

    def terminal_value(self):
        return self.solvers[1].advectee.get()

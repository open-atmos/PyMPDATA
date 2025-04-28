from functools import cached_property, lru_cache
from types import SimpleNamespace

import numba
import numpy as np
from PyMPDATA_examples.utils.discretisation import discretised_analytical_solution

from PyMPDATA import Options, ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Extrapolated
from PyMPDATA.impl.enumerations import (
    ARG_FOCUS,
    INNER,
    META_AND_DATA_DATA,
    META_AND_DATA_META,
    OUTER,
    SIGN_RIGHT,
)
from PyMPDATA.impl.traversals_common import make_fill_halos_loop


# pylint: disable=too-few-public-methods, too-many-lines, global-statement, abstract-method
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


_t = np.nan


@lru_cache()
# pylint: disable=too-many-arguments
def _make_scalar_custom(
    dim, eps, ats, set_value, halo, dtype, jit_flags, data, inner_or_outer
):
    @numba.njit(**jit_flags)
    def impl(focus_psi, span, sign):
        focus = focus_psi[0]
        i = min(max(0, focus[inner_or_outer] - halo), span - 1)
        if sign == SIGN_RIGHT:
            edg = span + halo - 1 - focus_psi[ARG_FOCUS][dim]
            den = ats(*focus_psi, edg - 1) - ats(*focus_psi, edg - 2)
            nom = ats(*focus_psi, edg) - ats(*focus_psi, edg - 1)
            cnst = nom / den if abs(den) > eps else 0
            return max(
                ats(*focus_psi, -1)
                + (ats(*focus_psi, -1) - ats(*focus_psi, -2)) * cnst,
                0,
            )
        return data[i]

    if dtype == complex:

        @numba.njit(**jit_flags)
        def fill_halos_scalar(psi, span, sign):
            return complex(
                impl(
                    (psi[META_AND_DATA_META], psi[META_AND_DATA_DATA].real), span, sign
                ),
                impl(
                    (psi[META_AND_DATA_META], psi[META_AND_DATA_DATA].imag), span, sign
                ),
            )

    else:

        @numba.njit(**jit_flags)
        def fill_halos_scalar(psi, span, sign):
            return impl(psi, span, sign)

    return make_fill_halos_loop(jit_flags, set_value, fill_halos_scalar)


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
        self.step_number = 0
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

    def add_half_rhs(self):
        pass

    def free_boundary(self, t):
        pass

    def step(self):
        global _t
        _t = self.settings.T - (self.step_number + 0.5) * self.dt
        self.add_half_rhs()
        self.free_boundary(t=_t)
        self.solver.advance(1, self.mu_coeff)
        self.add_half_rhs()
        self.free_boundary(t=_t)
        self.step_number += 1


class _Asian(Simulation):
    @cached_property
    def payoff_2d(self):
        return np.repeat([self.payoff], self.nx, axis=0)


class AsianArithmetic(_Asian):
    @property
    def a_dim_advector(self):
        a_dim_advector = np.zeros((self.nx, self.ny + 1))
        for i in range(self.ny + 1):
            a_dim_advector[:, i] = -self.dt / self.dy * self.S / self.settings.T
        return a_dim_advector

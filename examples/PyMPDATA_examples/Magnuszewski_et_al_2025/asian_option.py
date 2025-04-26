from abc import abstractmethod
from functools import cached_property, lru_cache
from types import SimpleNamespace

import numba
import numpy as np
import PyMPDATA_examples.utils.financial_formulae.asian_option as asian_analytic
from PyMPDATA_examples.utils.discretisation import discretised_analytical_solution
from PyMPDATA_examples.utils.financial_formulae import (
    Bjerksund_and_Stensland_1993,
    Black_Scholes_1973,
)

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

OPTIONS = {
    "n_iters": 3,
    "infinite_gauge": False,
    "nonoscillatory": True,
    "divergent_flow": False,
    "third_order_terms": False,
    "non_zero_mu_coeff": True,
}


class Settings:
    params = SimpleNamespace(
        T=1,
        sgma=0.1,
        r=0.1,
        K=100,
    )

    S_min = 50
    S_max = 200

    def __init__(self, T, sgma, r, K, S_min, S_max):
        self.params.T = T
        self.params.sgma = sgma
        self.params.r = r
        self.params.K = K
        self.S_min = S_min
        self.S_max = S_max

    def payoff(self, A: np.ndarray, da: np.float32 = 0, variant="call"):
        def call(x):
            return np.maximum(0, x - self.params.K)

        def put(x):
            return np.maximum(0, self.params.K - x)

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
            # TODO: reuse the code from Extrapolated (here copy & paste!)
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


class KemnaVorstBoundaryCondition(Extrapolated):
    """eq. (12) in [Kemna & Vorst (1990)](TODO) but without discounting
    which is embedded in the advectee definition (and applied to the initial condition)
    """

    def __init__(self, *, data_left, dim):
        super().__init__(dim=dim)
        self.data = tuple(data_left)
        self.inner_or_outer = (INNER, OUTER)[dim]

    def make_scalar(self, indexers, halo, dtype, jit_flags, dimension_index):
        return _make_scalar_custom(
            self.dim,
            self.eps,
            indexers.ats[dimension_index],
            indexers.set,
            halo,
            dtype,
            jit_flags,
            self.data,
            self.inner_or_outer,
        )


class Simulation:
    def __init__(self, settings, *, nx, ny, nt, variant="call"):
        self.nx = nx
        self.nt = nt
        self.settings = settings

        self.ny = ny
        self.dt = settings.params.T / self.nt
        log_s_min = np.log(settings.S_min)
        log_s_max = np.log(settings.S_max)
        self.S = np.exp(np.linspace(log_s_min, log_s_max, self.nx))
        self.A, self.dy = np.linspace(
            0, settings.S_max, self.ny, retstep=True, endpoint=True
        )
        self.dx = (log_s_max - log_s_min) / self.nx
        # self.dy =

        self.settings = settings
        self.step_number = 0

        sigma_squared = pow(settings.params.sgma, 2)
        courant_number_x = (
            -(0.5 * sigma_squared - settings.params.r) * (-self.dt) / self.dx
        )

        self.l2 = self.dx * self.dx / sigma_squared / self.dt

        self.mu_coeff = (0.5 / self.l2, 0)

        # print(f"{self.l2=}")

        assert (
            self.l2 > 2
        ), f"Lambda squared should be more than 2 for stability {self.l2}"

        # print(f"dx: {self.dx}, dy: {self.dy}, dt: {self.dt}")

        self.payoff = settings.payoff(A=self.A, da=self.dy, variant=variant)

        options = Options(**OPTIONS)
        stepper = Stepper(options=options, n_dims=2)

        x_dim_advector = np.full(
            (self.nx + 1, self.ny),
            courant_number_x,
            dtype=options.dtype,
        )
        # print(
        #     f"CFL {np.max(np.abs(self.a_dim_advector)) + np.max(np.abs(x_dim_advector))}"
        # )
        assert (
            np.max(np.abs(self.a_dim_advector)) + np.max(np.abs(x_dim_advector)) < 1
        ), f"CFL condition not met {np.max(np.abs(self.a_dim_advector)) + np.max(np.abs(x_dim_advector))}"

        courant_x = np.max(np.abs(x_dim_advector))
        courant_y = np.max(np.abs(self.a_dim_advector))
        # print(f"{courant_x=}, {courant_y=}")

        # print(f"{x_dim_advector.shape=}, {self.a_dim_advector.shape=}")

        self.solver = Solver(
            stepper=stepper,
            advectee=ScalarField(
                self.payoff_2d.astype(dtype=options.dtype)
                * np.exp(-self.settings.params.r * self.settings.params.T),
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
        raise NotImplementedError()

    def add_half_rhs(self):
        raise NotImplementedError()

    def free_boundary(self, t):
        raise NotImplementedError()

    def step(self):
        global _t
        _t = self.settings.params.T - (self.step_number + 0.5) * self.dt

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

    def free_boundary(self, t):
        pass

    @cached_property
    def boundary_conditions(self):
        return (
            # KemnaVorstBoundaryCondition(
            #     data_left=self.payoff_2d[0, :]
            #     * np.exp(-self.settings.params.r * self.settings.params.T),
            #     dim=OUTER,
            # ),
            Extrapolated(OUTER),
            Extrapolated(INNER),
        )


class AsianArithmetic(_Asian):
    @property
    def a_dim_advector(self):
        a_dim_advector = np.zeros((self.nx, self.ny + 1))
        # y_edg = np.log(self.S[0]) + np.arange(self.ny + 1) * self.dy
        # A_edg = np.exp(y_edg - self.dy / 2)
        for i in range(self.ny + 1):
            a_dim_advector[:, i] = -self.dt / self.dy * self.S / self.settings.params.T
        return a_dim_advector

    def add_half_rhs(self):
        pass
        # psi = self.solver.advectee.get()
        # self.rhs[:] = -psi / self.settings.params.T # todo male t
        # psi[:] = np.maximum(0, psi + -self.dt * self.rhs / 2)


class AsianGeometric(_Asian):
    @property
    def a_dim_advector(self):
        a_dim_advector = np.zeros((self.nx, self.ny + 1))
        # y_edg = np.log(self.S[0]) + np.arange(self.ny + 1) * self.dy
        # A_edg = np.exp(y_edg - self.dy / 2)
        A_edg = np.arange(self.ny + 1) * self.dy
        for i in range(self.ny + 1):
            a_dim_advector[:, i] = (
                -self.dt
                / self.dy
                * np.log(self.S)
                * (A_edg[i] - self.dy / 2)
                / self.settings.params.T
            )
        return a_dim_advector

    def add_half_rhs(self):
        pass
        # psi = self.solver.advectee.get()
        # self.rhs[:] = psi / self.settings.params.T * np.log(self.S)
        # psi[:] = np.maximum(0, psi + -self.dt * self.rhs / 2)


class European(Simulation):
    @cached_property
    def a_dim_advector(self):
        return np.zeros((self.nx, self.ny + 1))

    def add_half_rhs(self):
        pass

    @cached_property
    def payoff_2d(self):
        return np.repeat(self.payoff.reshape(self.nx, 1), self.ny, axis=1)

    def free_boundary(self, t):
        pass

    @cached_property
    def boundary_conditions(self):
        return (
            Extrapolated(OUTER),
            Extrapolated(INNER),
        )


class American(European):
    def free_boundary(self, t):
        psi = self.solver.advectee.get()
        psi[:] = np.maximum(psi, self.payoff_2d / np.exp(self.settings.params.r * t))


def plot_solution(
    settings,
    frame_index,
    ax,
    history,
    S_linspace,
    arithmetic_by_mc,
    option_type: str,
    variant,
    A_space,
):
    params = {
        k: v for k, v in settings.params.__dict__.items() if not k.startswith("K")
    }
    if variant == "call":
        BS_price_func = Black_Scholes_1973.c_euro
        Amer_price_func = Bjerksund_and_Stensland_1993.c_amer
        geometric_price_func = asian_analytic.geometric_asian_average_price_c
    else:
        BS_price_func = Black_Scholes_1973.p_euro
        Amer_price_func = Bjerksund_and_Stensland_1993.p_amer
        geometric_price_func = asian_analytic.geometric_asian_average_price_p

    ax.plot(
        S_linspace,
        (
            BS_price_func(
                S=S_linspace, K=settings.params.K, **params, b=settings.params.r
            )
        ),
        label="European analytic (Black-Scholes '73)",
        linestyle="--",
        alpha=0.5,
    )
    # ax.plot(
    #     S_linspace,
    #     (
    #         Amer_price_func(
    #             S=S_linspace, K=settings.params.K, **params, b=settings.params.r
    #         )
    #     ),
    #     label="American analytic (Bjerksund & Stensland '93)", linestyle='--'
    # )
    ax.plot(
        S_linspace,
        (
            geometric_price_func(
                S=S_linspace, K=settings.params.K, **params, dividend_yield=0
            )
        ),
        label="Asian analytic (geometric, Kemna & Vorst 1990)",
        alpha=0.5,
        linestyle="--",
    )
    ax.plot(
        S_linspace,
        arithmetic_by_mc,
        label="Asian arithmetic by Monte-Carlo",
        linestyle=":",
    )
    ax.plot(
        S_linspace,
        history[frame_index][:, 0],
        label=f"MPDATA solution ({option_type})",
        marker=".",
    )
    ax.plot(
        A_space,
        history[0][0, :],
        label=f"Terminal condition (discounted payoff)",
        marker=".",
    )
    ax.legend(loc="upper right")
    ax.grid()
    minmax = (np.amin(history[0]), np.amax(history[0]))
    span = minmax[1] - minmax[0]
    ax.set_ylim(minmax[0] - 0.05 * span, minmax[1] + 0.25 * span)
    ax.set_title(f"instrument parameters: {settings.params.__dict__}")
    # ax.set_xlabel("underlying S(t=0)=A(t=0) (and A(T) for terminal condition)")
    ax.set_xlabel("average at t=T for payoff, spot at t=0 for other datasets")
    ax.set_ylabel("value")


def plot_difference_arithmetic(
    settings, frame_index, ax, history, S_linspace, arithmetic_by_mc, option_type: str
):
    ax.plot(
        S_linspace,
        abs(arithmetic_by_mc - history[frame_index][:, 0]) / arithmetic_by_mc,
        label="Difference in price of Asian arithmetic option MPDATA vs MC",
    )
    ax.legend(loc="upper right")
    ax.grid()
    ax.set_title(f"instrument parameters: {settings.params.__dict__}")
    ax.set_xlabel("underlying S(t=0)=A(t=0) (and A(T) for terminal condition)")
    ax.set_ylabel("option price error")
    ax.set_yscale("log")
    ax.set_ylim(0, 100)

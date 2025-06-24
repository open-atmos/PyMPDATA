"""
Helper functions to run two different implementations of the diffusion equation with spatial dependence.
"""

import dataclasses
import logging
from typing import Dict, Tuple, Any

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

# from pde import PDE, CartesianGrid, MemoryStorage, ScalarField, plot_kymograph
import pde as py_pde

from PyMPDATA import Options, ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Constant


@dataclasses.dataclass(frozen=True)
class SimulationArgs:
    """Dataclass to hold simulation arguments."""

    grid_bounds: Tuple[float, float]
    grid_points: int
    initial_value: float
    sim_time: float
    dt: float


@dataclasses.dataclass(frozen=True)
class SimulationResult:
    """Dataclass to hold simulation results, and additional produced plots."""

    kymograph_result: np.ndarray
    figures: Dict[str, matplotlib.figure.Figure] = dataclasses.field(
        default_factory=dict
    )
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)


def py_pde_solution(args: SimulationArgs):
    """Runs the simulation using pyPDE."""

    term_1 = "(1.01 + tanh(x)) * laplace(c)"
    term_2 = "dot(gradient(1.01 + tanh(x)), gradient(c))"
    eq = py_pde.PDE({"c": f"{term_1} + {term_2}"}, bc={"value": 0})

    grid = py_pde.CartesianGrid([args.grid_bounds], args.grid_points)
    field = py_pde.ScalarField(grid, args.initial_value)

    storage = py_pde.MemoryStorage()
    result = eq.solve(field, args.sim_time, dt=args.dt, tracker=storage.tracker(1))

    return SimulationResult(
        kymograph_result=np.array(storage.data),
        extra={"final_result": result, "storage": storage},
    )


def pympdata_solution(args: SimulationArgs) -> SimulationResult:
    """Runs the simulation using PyMPDATA."""

    xmin, xmax = args.grid_bounds
    dx = (xmax - xmin) / args.grid_points
    x = np.linspace(xmin + dx / 2, xmax - dx / 2, args.grid_points)

    n_steps = int(args.sim_time / args.dt)

    D_field = 1.01 + np.tanh(x)

    # initial condition - uniform field (to match py-pde reference exactly)
    c0 = np.full(
        args.grid_points, args.initial_value
    )  # Uniform concentration everywhere

    # ── build a Solver with native heterogeneous diffusion ───────────────────────────
    opts = Options(
        n_iters=10,  # more MPDATA iterations → sharper features
        non_zero_mu_coeff=True,
        heterogeneous_diffusion=True,  # Enable native heterogeneous diffusion
    )

    # Set up fields with proper boundary conditions
    advectee = ScalarField(
        data=c0, halo=opts.n_halo, boundary_conditions=(Constant(0.0),)
    )
    advector = VectorField(
        data=(np.zeros(args.grid_points + 1),),
        halo=opts.n_halo,
        boundary_conditions=(Constant(0.0),),
    )
    diffusivity_field = ScalarField(
        data=D_field, halo=opts.n_halo, boundary_conditions=(Constant(0.0),)
    )

    stepper = Stepper(options=opts, grid=(args.grid_points,))
    solver = Solver(
        stepper=stepper,
        advectee=advectee,
        advector=advector,
        diffusivity_field=diffusivity_field,
    )

    # ── march & record for kymograph ──────────────────────────────────────────────
    logging.info("Starting heterogeneous diffusion simulation...")
    logging.info(
        "Using native PyMPDATA implementation (should be ~3x faster than Strang splitting)"
    )

    kymo = np.empty((n_steps + 1, args.grid_points))
    kymo[0] = solver.advectee.get()

    # Use stronger mu_coeff for more realistic long-time evolution
    mu_coeff = 0.05  # Increased to get more decay over time

    logging.info(f"Diffusivity range: {D_field.min():.3f} to {D_field.max():.3f}")
    logging.info(f"Using balanced mu coefficient: {mu_coeff:.6f}")

    for i in range(1, n_steps + 1):
        if i % 10000 == 0:
            logging.info(f"At step {i}/{n_steps}")

        # Single call per timestep (vs 3 calls in Strang splitting!)
        solver.advance(n_steps=1, mu_coeff=(mu_coeff,))
        kymo[i] = solver.advectee.get()

    logging.info("Simulation completed!")

    res_kymo = np.empty((int(args.sim_time), args.grid_points))
    interval = int(1 / args.dt)

    for i in range(int(args.sim_time)):
        step_data = kymo[i * interval + 1 : (i + 1) * interval + 1]
        res_kymo[i] = step_data[step_data.shape[0] // 2]

    res_kymo = np.concat((kymo[0:1], res_kymo), axis=0)

    # ── plot ───────────────────────────────────────────────────────────────────────
    T = np.linspace(0, args.sim_time, int(args.sim_time) + 1)
    X, Tgrid = np.meshgrid(x, T)

    figs = {}

    fig1, ax1 = plt.subplots(figsize=(10, 8))
    pcm = ax1.pcolormesh(X, Tgrid, res_kymo, shading="auto")
    ax1.set_xlabel("x")
    ax1.set_ylabel("Time")
    figs["kymograph"] = fig1
    fig1.colorbar(pcm, ax=ax1)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(x, D_field, "b-", linewidth=2)
    ax2.set_xlabel("x")
    ax2.set_ylabel("D(x)")
    ax2.set_title("Heterogeneous Diffusivity")
    ax2.grid(True, alpha=0.3)
    figs["diffusivity_profile"] = fig2
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    ax3.plot(x, kymo[0], "k--", alpha=0.7, label="t=0")
    ax3.plot(x, kymo[-1], "r-", label=f"t={args.sim_time}")
    ax3.set_xlabel("x")
    ax3.set_ylabel("c(x)")
    ax3.set_title("Initial vs Final")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    figs["initial_vs_final"] = fig3
    plt.close(fig3)

    # ── Summary statistics ─────────────────────────────────────────────────────────
    logging.info(
        f"Mass conservation: initial={kymo[0].sum():.6f}, final={kymo[-1].sum():.6f}"
    )
    logging.info(
        f"Relative mass change: {abs(kymo[-1].sum() - kymo[0].sum()) / kymo[0].sum() * 100:.2e}%"
    )

    return SimulationResult(kymograph_result=res_kymo, figures=figs)

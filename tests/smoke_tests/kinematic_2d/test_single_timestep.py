# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from PyMPDATA_examples.utils import nondivergent_vector_field_2d

from PyMPDATA import Options, ScalarField, Solver, Stepper
from PyMPDATA.boundary_conditions import Periodic

GRID = (75, 75)
SIZE = (1500, 1500)
TIMESTEP = 1
RHOD_W_MAX = 0.6


def stream_function(x_01, z_01):
    x_span = SIZE[0]
    return (
        -RHOD_W_MAX * x_span / np.pi * np.sin(np.pi * z_01) * np.cos(2 * np.pi * x_01)
    )


def rhod_of_z(arg):
    return 1 - arg * 1e-4


RHOD = np.repeat(
    rhod_of_z((np.arange(GRID[1]) + 1 / 2) / GRID[1]).reshape((1, GRID[1])),
    GRID[0],
    axis=0,
)

VALUES = {"th": np.full(GRID, 300), "qv": np.full(GRID, 0.001)}


@pytest.mark.parametrize(
    "options",
    (
        Options(n_iters=1),
        Options(n_iters=2),
        Options(n_iters=2, nonoscillatory=True),
        Options(n_iters=3, nonoscillatory=True),
        Options(n_iters=2, nonoscillatory=True, infinite_gauge=True),
        Options(nonoscillatory=True, infinite_gauge=True, third_order_terms=True),
        Options(nonoscillatory=False, infinite_gauge=True),
        Options(nonoscillatory=False, third_order_terms=True),
        Options(nonoscillatory=False, infinite_gauge=True, third_order_terms=True),
    ),
)
def test_single_timestep(options):
    # Arrange
    stepper = Stepper(options=options, grid=GRID, non_unit_g_factor=True)
    advector = nondivergent_vector_field_2d(
        GRID, SIZE, TIMESTEP, stream_function, options.n_halo
    )
    g_factor = ScalarField(
        RHOD.astype(dtype=options.dtype),
        halo=options.n_halo,
        boundary_conditions=(Periodic(), Periodic()),
    )
    mpdatas = {}
    for key, value in VALUES.items():
        advectee = ScalarField(
            np.full(GRID, value, dtype=options.dtype),
            halo=options.n_halo,
            boundary_conditions=(Periodic(), Periodic()),
        )
        mpdatas[key] = Solver(
            stepper=stepper, advectee=advectee, advector=advector, g_factor=g_factor
        )

    # Act
    for mpdata in mpdatas.values():
        mpdata.advance(n_steps=1)

    # Assert
    for value in mpdatas.values():
        assert np.isfinite(value.advectee.get()).all()

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numba
import numpy as np
import pytest
from matplotlib import pyplot

from PyMPDATA import Options, ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Periodic

from .fixtures import num_threads

assert hasattr(num_threads, "_pytestfixturefunction")


GRID = (126, 101)

TIMESTEP = 0.1
DX = 1
DY = 1
OMEGA = 0.1
H = 4.0
H_0 = 1

RADIUS = 0.15 * GRID[0] * DX
X_0 = int(0.5 * GRID[0]) * DX
Y_0 = int(0.2 * GRID[0]) * DY

X_C = 0.5 * GRID[0] * DX
Y_C = 0.5 * GRID[1] * DY
COURANT = (-0.5, 0.25)


class Settings:
    def __init__(self, n_rotations=6):
        self.n_rotations = n_rotations

    @property
    def timestep(self):
        return TIMESTEP

    @property
    def n_steps(self):
        return int(100 * self.n_rotations)

    @property
    def size(self):
        return self.xrange[1], self.yrange[1]

    @property
    def xrange(self):
        return 0, GRID[0] * DX

    @property
    def yrange(self):
        return 0, GRID[1] * DY

    @property
    def grid(self):
        return GRID

    @staticmethod
    def pdf(coord_x, coord_y):
        tmp = (coord_x - X_0) ** 2 + (coord_y - Y_0) ** 2
        return H_0 + np.where(
            # if
            tmp - RADIUS**2 <= 0,
            # then
            H - np.sqrt(tmp / (RADIUS / H) ** 2),
            # else
            0.0,
        )


def from_pdf_2d(pdf, xrange, yrange, gridsize):
    psi = np.empty(gridsize)
    delta_x = (xrange[1] - xrange[0]) / gridsize[0]
    delta_y = (yrange[1] - yrange[0]) / gridsize[1]
    for i in range(gridsize[0]):
        for j in range(gridsize[1]):
            psi[i, j] = pdf(
                xrange[0] + delta_x * (i + 0.5), yrange[0] + delta_y * (j + 0.5)
            )
    coord_x = np.linspace(xrange[0] + delta_x / 2, xrange[1] - delta_x / 2, gridsize[0])
    coord_y = np.linspace(yrange[0] + delta_y / 2, yrange[1] - delta_y / 2, gridsize[1])
    return coord_x, coord_y, psi


@pytest.mark.parametrize(
    "options",
    [
        Options(n_iters=1),
        Options(n_iters=2),
        Options(n_iters=3, infinite_gauge=True),
        Options(n_iters=2, infinite_gauge=True, nonoscillatory=True),
        # Options(n_iters=3, infinite_gauge=False, third_order_terms=True),
        # Options(n_iters=3, infinite_gauge=True, third_order_terms=True, nonoscillatory=True),
    ],
)
@pytest.mark.parametrize("grid_static_str", ("static", "dynamic"))
# pylint: disable-next=redefined-outer-name
def test_timing_2d(benchmark, options, grid_static_str, num_threads, plot=False):
    if grid_static_str == "static":
        grid_static = True
    elif grid_static_str == "dynamic":
        grid_static = False
    else:
        raise ValueError()

    numba.set_num_threads(num_threads)

    settings = Settings(n_rotations=6)
    _, __, psi = from_pdf_2d(
        settings.pdf,
        xrange=settings.xrange,
        yrange=settings.yrange,
        gridsize=settings.grid,
    )

    advectee = ScalarField(
        data=psi.astype(dtype=options.dtype),
        halo=options.n_halo,
        boundary_conditions=(Periodic(), Periodic()),
    )

    advector = VectorField(
        data=(
            np.full(
                (advectee.grid[0] + 1, advectee.grid[1]),
                COURANT[0],
                dtype=options.dtype,
            ),
            np.full(
                (advectee.grid[0], advectee.grid[1] + 1),
                COURANT[1],
                dtype=options.dtype,
            ),
        ),
        halo=options.n_halo,
        boundary_conditions=(Periodic(), Periodic()),
    )

    if grid_static:
        stepper = Stepper(options=options, grid=psi.shape)
    else:
        stepper = Stepper(options=options, n_dims=2)
    solver = Solver(stepper=stepper, advectee=advectee, advector=advector)

    def set_z():
        solver.advectee.get()[:] = psi

    benchmark.pedantic(
        solver.advance, (settings.n_steps,), setup=set_z, warmup_rounds=1, rounds=3
    )

    if options.n_iters == 1 or options.nonoscillatory:
        np.testing.assert_almost_equal(np.amin(solver.advectee.get()), H_0)
    assert np.amax(solver.advectee.get()) < 10 * H

    if plot:
        pyplot.imshow(solver.advectee.get())
        pyplot.colorbar()
        pyplot.show()

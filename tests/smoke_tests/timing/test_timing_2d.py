# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import numba
from matplotlib import pyplot
import pytest
from PyMPDATA import ScalarField, VectorField, Options, Solver, Stepper
from PyMPDATA.boundary_conditions import Periodic

from .fixtures import num_threads
assert hasattr(num_threads, '_pytestfixturefunction')


GRID = (126, 101)

TIMESTEP = .1
DX = 1
DY = 1
OMEGA = .1
H = 4.
H_0 = 1

r = .15 * GRID[0] * DX
X_0 = int(.5 * GRID[0]) * DX
Y_0 = int(.2 * GRID[0]) * DY

X_C = .5 * GRID[0] * DX
Y_C = .5 * GRID[1] * DY


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
    def pdf(x, y):
        tmp = (x - X_0) ** 2 + (y - Y_0) ** 2
        return H_0 + np.where(
            # if
            tmp - r ** 2 <= 0,
            # then
            H - np.sqrt(tmp / (r / H) ** 2),
            # else
            0.
        )


def from_pdf_2d(pdf, xrange, yrange, gridsize):
    z = np.empty(gridsize)
    dx, dy = (xrange[1] - xrange[0]) / gridsize[0], (yrange[1] - yrange[0]) / gridsize[1]
    for i in range(gridsize[0]):
        for j in range(gridsize[1]):
            z[i, j] = pdf(
                xrange[0] + dx * (i + .5),
                yrange[0] + dy * (j + .5)
            )
    x = np.linspace(xrange[0] + dx / 2, xrange[1] - dx / 2, gridsize[0])
    y = np.linspace(yrange[0] + dy / 2, yrange[1] - dy / 2, gridsize[1])
    return x, y, z


@pytest.mark.parametrize("options", [
    Options(n_iters=1),
    Options(n_iters=2),
    Options(n_iters=3, infinite_gauge=True),
    Options(n_iters=2, infinite_gauge=True, nonoscillatory=True),
    # Options(n_iters=3, infinite_gauge=False, third_order_terms=True),
    # Options(n_iters=3, infinite_gauge=True, third_order_terms=True, nonoscillatory=True),
])
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
    _, __, z = from_pdf_2d(settings.pdf,
                           xrange=settings.xrange,
                           yrange=settings.yrange,
                           gridsize=settings.grid)

    courant = (-.5, .25)
    grid = z.shape
    advector_data = [
        np.full((grid[0] + 1, grid[1]), courant[0], dtype=options.dtype),
        np.full((grid[0], grid[1] + 1), courant[1], dtype=options.dtype)
    ]
    advector = VectorField(advector_data, halo=options.n_halo,
                           boundary_conditions=(Periodic(), Periodic()))
    advectee = ScalarField(data=z.astype(dtype=options.dtype), halo=options.n_halo,
                           boundary_conditions=(Periodic(), Periodic()))
    if grid_static:
        stepper = Stepper(options=options, grid=grid)
    else:
        stepper = Stepper(options=options, n_dims=2)
    solver = Solver(stepper=stepper, advectee=advectee, advector=advector)

    def set_z():
        solver.advectee.get()[:] = z

    benchmark.pedantic(solver.advance, (settings.n_steps,), setup=set_z, warmup_rounds=1, rounds=3)

    if options.n_iters == 1 or options.nonoscillatory:
        np.testing.assert_almost_equal(np.amin(solver.advectee.get()), H_0)
    assert np.amax(solver.advectee.get()) < 10 * H

    if plot:
        pyplot.imshow(solver.advectee.get())
        pyplot.colorbar()
        pyplot.show()

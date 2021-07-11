from PyMPDATA import ScalarField, VectorField, PeriodicBoundaryCondition, Solver, Stepper
from PyMPDATA.options import Options
import numpy as np
import numba
from matplotlib import pyplot
import pytest

grid = (126, 101)

dt = .1
dx = 1
dy = 1
omega = .1
h = 4.
h0 = 1

r = .15 * grid[0] * dx
x0 = int(.5 * grid[0]) * dx
y0 = int(.2 * grid[0]) * dy

xc = .5 * grid[0] * dx
yc = .5 * grid[1] * dy


class Settings:
    def __init__(self, n_rotations=6):
        self.n_rotations = n_rotations

    @property
    def dt(self):
        return dt

    @property
    def nt(self):
        return int(100 * self.n_rotations)

    @property
    def size(self):
        return self.xrange[1], self.yrange[1]

    @property
    def xrange(self):
        return 0, grid[0] * dx

    @property
    def yrange(self):
        return 0, grid[1] * dy

    @property
    def grid(self):
        return grid

    @staticmethod
    @numba.njit()
    def pdf(x, y):
        tmp = (x-x0)**2 + (y-y0)**2
        return h0 + np.where(
            # if
            tmp - r**2 <= 0,
            # then
            h - np.sqrt(tmp / (r/h)**2),
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


concurrency_str = ("threads", "serial")
try:
    numba.parfors.parfor.ensure_parallel_support()
except numba.core.errors.UnsupportedParforsError:
    concurrency_str = ("serial")


@pytest.mark.parametrize("options", [
    Options(n_iters=1),
    Options(n_iters=2),
    Options(n_iters=3, infinite_gauge=True),
    Options(n_iters=2, infinite_gauge=True, flux_corrected_transport=True),
    Options(n_iters=3, infinite_gauge=False, third_order_terms=True),
    Options(n_iters=3, infinite_gauge=True, third_order_terms=True, flux_corrected_transport=True),
])
@pytest.mark.parametrize("dtype", (np.float64,))
@pytest.mark.parametrize("grid_static_str", ("static", "dynamic"))
@pytest.mark.parametrize("concurrency_str", concurrency_str)
def test_timing_2d(benchmark, options, dtype, grid_static_str, concurrency_str, plot=False):
    if grid_static_str == "static":
        grid_static = True
    elif grid_static_str == "dynamic":
        grid_static = False
    else:
        raise ValueError()

    if concurrency_str == "serial":
        numba.set_num_threads(1)
    else:
        numba.set_num_threads(numba.config.NUMBA_NUM_THREADS)

    settings = Settings(n_rotations=6)
    _, __, z = from_pdf_2d(settings.pdf, xrange=settings.xrange, yrange=settings.yrange, gridsize=settings.grid)

    C = (-.5, .25)
    grid = z.shape
    advector_data = [
        np.full((grid[0] + 1, grid[1]), C[0], dtype=options.dtype),
        np.full((grid[0], grid[1] + 1), C[1], dtype=options.dtype)
    ]
    advector = VectorField(advector_data, halo=options.n_halo,
                     boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition()))
    advectee = ScalarField(data=z.astype(dtype=options.dtype), halo=options.n_halo,
                        boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition()))
    if grid_static:
        stepper = Stepper(options=options, grid=grid)
    else:
        stepper = Stepper(options=options, n_dims=2)
    solver = Solver(stepper=stepper, advectee=advectee, advector=advector)

    def set_z():
        solver.advectee.get()[:] = z

    benchmark.pedantic(solver.advance, (settings.nt,), setup=set_z, warmup_rounds=1, rounds=3)

    if options.n_iters == 1 or options.flux_corrected_transport:
        np.testing.assert_almost_equal(np.amin(solver.advectee.get()), h0)
    assert np.amax(solver.advectee.get()) < 10 * h

    if plot:
        pyplot.imshow(solver.advectee.get())
        pyplot.colorbar()
        pyplot.show()

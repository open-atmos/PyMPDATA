from MPyDATA.factories import Factories
from MPyDATA.options import Options
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


class Setup:
    def __init__(self, n_rotations=6):
        self.n_rotations = n_rotations

    @property
    def dt(self):
        return dt

    @property
    def nt(self):
        return int(300 * self.n_rotations)

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


# Options(n_iters=2, infinite_gauge=True, flux_corrected_transport=True),  # TODO!
# TODO: threading_layer
@pytest.mark.parametrize("options", [
    Options(n_iters=1),
    Options(n_iters=2),
    Options(n_iters=3, infinite_gauge=True),
    Options(n_iters=2, flux_corrected_transport=True),
])
@pytest.mark.parametrize("dtype", (np.float64,))
@pytest.mark.parametrize("grid_static_str", ("static", "dynamic"))
@pytest.mark.parametrize("concurrency_str", ("serial", "threads"))
def test_timing_2d(benchmark, options, dtype, grid_static_str, concurrency_str):
    if grid_static_str == "static":
        grid_static = True
    elif grid_static_str == "dynamic":
        grid_static = False
    else:
        raise ValueError()

    if concurrency_str == "serial":
        numba.set_num_threads(1)
    elif concurrency_str == "threads":
        numba.config.THREADING_LAYER = 'threads'
        numba.set_num_threads(numba.config.NUMBA_NUM_THREADS // 2)
    else:
        raise ValueError()

    setup = Setup(n_rotations=6)
    _, __, z = from_pdf_2d(setup.pdf, xrange=setup.xrange, yrange=setup.yrange, gridsize=setup.grid)
    mpdata = Factories.constant_2d(data=z, C=(-.5, .25), options=options, grid_static=grid_static)

    def set_z():
        mpdata.curr.get()[:] = z

    benchmark.pedantic(mpdata.advance, (setup.nt,), setup=set_z, warmup_rounds=1, rounds=3)
    state = mpdata.curr.get()

    print(np.amin(state), np.amax(state))
    if options.n_iters == 1:
        assert np.amin(state) >= h0
    assert np.amax(state) < 10 * h

    if False:
        pyplot.imshow(state)
        pyplot.colorbar()
        pyplot.show()


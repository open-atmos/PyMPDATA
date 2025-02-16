import warnings

import numpy as np
from matplotlib import pyplot, colors
import pytest
from numba import NumbaExperimentalFeatureWarning

from PyMPDATA import ScalarField, VectorField, Options
from PyMPDATA.boundary_conditions import Extrapolated, Constant
from PyMPDATA.impl.enumerations import MAX_DIM_NUM
from PyMPDATA.impl.traversals import Traversals
from PyMPDATA.impl.field import Field

JIT_FLAGS = Options().jit_flags


def quick_look(field: Field, plot:bool):
    halo = field.halo
    grid = field.grid
    pyplot.title(f"{grid=} {halo=}")
    if isinstance(field, ScalarField):
        norm = colors.Normalize(
            vmin=np.amin(field.get()),
            vmax=np.amax(field.get())
        )
        pyplot.imshow(
            X=field.data.T,
            origin='lower',
            extent=(
                -halo, grid[0] + halo,
                -halo, grid[1] + halo
            ),
            cmap='gray',
            norm=norm,
        )
        pyplot.colorbar()
    pyplot.hlines(y=range(-halo, grid[1] + 1 + halo), xmin=-halo, xmax=grid[0] + halo, color='r', linewidth=.5)
    pyplot.vlines(x=range(-halo, grid[0] + 1 + halo), ymin=-halo, ymax=grid[1] + halo, color='r', linewidth=.5)
    pyplot.hlines(y=range(grid[1] + 1), xmin=0, xmax=grid[0], color='r', linewidth=3)
    pyplot.vlines(x=range(grid[0] + 1), ymin=0, ymax=grid[1], color='r', linewidth=3)
    for i, xy in enumerate(('x', 'y')):
        getattr(pyplot, f"{xy}ticks")(
            np.linspace(-halo + .5, grid[i] + halo - .5, grid[i] + 2 * halo)
        )
    pyplot.xlabel('x/dx')
    pyplot.ylabel('y/dy')
    pyplot.grid()
    if plot:
        pyplot.show()
    else:
        pyplot.clf()

def fill_halos(field: Field, traversals: Traversals, threads):
    field.assemble(traversals)
    meta_and_data, fill_halos = field.impl
    if isinstance(field, VectorField):
        meta_and_data = (
            meta_and_data[0],
            (meta_and_data[1], meta_and_data[2], meta_and_data[3]),
        )
    sut = traversals._code[{
        'ScalarField': "fill_halos_scalar",
        'VectorField': "fill_halos_vector"
    }[field.__class__.__name__]]  # pylint:disable=protected-access
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=NumbaExperimentalFeatureWarning)
        for thread_id in threads:
            sut(thread_id, *meta_and_data, fill_halos, traversals.data.buffer)

class TestBoundaryConditionExtrapolated2D:
    @staticmethod
    @pytest.mark.parametrize("n_threads", (1,2))
    @pytest.mark.parametrize("n_halo", (1,2))
    @pytest.mark.parametrize("bc", (
        (Extrapolated(0), Extrapolated(-1)),
        (Constant(0), Extrapolated(-1)),
        (Extrapolated(0), Constant(0)),
    ))
    def test_scalar_field(n_threads: int, n_halo: int, bc: tuple, plot=False):
        # arrange
        advectee = ScalarField(
            data=np.asarray([
                [0.,1.,2.],
                [3.,4.,5.]
            ]),
            boundary_conditions=bc,
            halo=n_halo
        )

        traversals = Traversals(
            grid=advectee.grid,
            halo=n_halo,
            jit_flags=JIT_FLAGS,
            n_threads=n_threads,
            left_first=tuple([True] * MAX_DIM_NUM),
            buffer_size=0,
        )

        # act / plot
        quick_look(advectee, plot)
        fill_halos(advectee, traversals, threads=range(n_threads))
        quick_look(advectee, plot)

        # assert
        assert np.isfinite(advectee.data).all()

    @staticmethod
    @pytest.mark.parametrize("n_threads", (1, 2))
    @pytest.mark.parametrize("n_halo", (1, 2))
    @pytest.mark.parametrize("bc", (
        (Extrapolated(0), Extrapolated(-1)),
        (Constant(0), Extrapolated(-1)),
        (Extrapolated(0), Constant(0)),
    ))
    def test_vector_field(n_threads: int, n_halo: int, bc:tuple, plot=False):
        # arrange
        advector = VectorField(
            data=(
                np.asarray([
                    [0., 1.],
                    [2., 3.],
                    [4., 5.]
                ]),
                np.asarray([
                    [0., 1., 2.],
                    [3., 4., 5.]
                ]),
            ),
            boundary_conditions=bc,
            halo=n_halo
        )

        traversals = Traversals(
            grid=advector.grid,
            halo=n_halo,
            jit_flags=JIT_FLAGS,
            n_threads=n_threads,
            left_first=tuple([True] * MAX_DIM_NUM),
            buffer_size=0,
        )

        # act / plot
        quick_look(advector, plot)
        fill_halos(advector, traversals, threads=range(n_threads))
        quick_look(advector, plot)

        # assert
        for component in advector.data:
            assert np.isfinite(component).all()
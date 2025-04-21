# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

import numpy as np
import pytest

from PyMPDATA import Options, ScalarField, VectorField
from PyMPDATA.boundary_conditions import Constant, Extrapolated
from PyMPDATA.impl.enumerations import MAX_DIM_NUM
from PyMPDATA.impl.traversals import Traversals
from tests.unit_tests.quick_look import quick_look

JIT_FLAGS = Options().jit_flags


class TestBoundaryConditionExtrapolated2D:
    @staticmethod
    @pytest.mark.parametrize("n_threads", (1, 2))
    @pytest.mark.parametrize("n_halo", (1, 2))
    @pytest.mark.parametrize(
        "boundary_conditions",
        (
            (Extrapolated(0), Extrapolated(-1)),
            (Constant(0), Extrapolated(-1)),
            (Extrapolated(0), Constant(0)),
        ),
    )
    def test_scalar_field(
        n_threads: int, n_halo: int, boundary_conditions: tuple, plot=False
    ):
        # arrange
        advectee = ScalarField(
            data=np.asarray([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
            boundary_conditions=boundary_conditions,
            halo=n_halo,
        )
        traversals = Traversals(
            grid=advectee.grid,
            halo=n_halo,
            jit_flags=JIT_FLAGS,
            n_threads=n_threads,
            left_first=tuple([True] * MAX_DIM_NUM),
            buffer_size=0,
        )
        advectee.assemble(traversals)

        # act / plot
        quick_look(advectee, plot)
        advectee._debug_fill_halos(  # pylint:disable=protected-access
            traversals, range(n_threads)
        )
        quick_look(advectee, plot)

        # assert
        assert np.isfinite(advectee.data).all()

    @staticmethod
    @pytest.mark.parametrize("n_threads", (1, 2))
    @pytest.mark.parametrize("n_halo", (1, 2))
    @pytest.mark.parametrize(
        "boundary_conditions",
        (
            (Extrapolated(0), Extrapolated(-1)),
            (Constant(0), Extrapolated(-1)),
            (Extrapolated(0), Constant(0)),
        ),
    )
    def test_vector_field(
        n_threads: int, n_halo: int, boundary_conditions: tuple, plot=False
    ):
        # arrange
        advector = VectorField(
            data=(
                np.asarray([[-1, 1.0], [2.0, 3.0], [4.0, 5.0]]),
                np.asarray([[-1.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
            ),
            boundary_conditions=boundary_conditions,
            halo=n_halo,
        )

        traversals = Traversals(
            grid=advector.grid,
            halo=n_halo,
            jit_flags=JIT_FLAGS,
            n_threads=n_threads,
            left_first=tuple([True] * MAX_DIM_NUM),
            buffer_size=0,
        )
        advector.assemble(traversals)

        # act / plot
        quick_look(advector, plot)
        advector._debug_fill_halos(  # pylint:disable=protected-access
            traversals, range(n_threads)
        )
        quick_look(advector, plot)

        # assert
        for component in advector.data:
            assert np.isfinite(component).all()

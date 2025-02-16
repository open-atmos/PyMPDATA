# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import warnings

import numpy as np
import pytest
from numba import NumbaExperimentalFeatureWarning

from PyMPDATA import Options, ScalarField, VectorField
from PyMPDATA.boundary_conditions import Constant, Extrapolated
from PyMPDATA.impl.enumerations import MAX_DIM_NUM
from PyMPDATA.impl.field import Field
from PyMPDATA.impl.traversals import Traversals
from tests.unit_tests.quick_look import quick_look

JIT_FLAGS = Options().jit_flags


def fill_halos(field: Field, traversals: Traversals, threads):
    field.assemble(traversals)
    meta_and_data, fill_halos_fun = field.impl
    if isinstance(field, VectorField):
        meta_and_data = (
            meta_and_data[0],
            (meta_and_data[1], meta_and_data[2], meta_and_data[3]),
        )
    sut = traversals._code[  # pylint:disable=protected-access
        {"ScalarField": "fill_halos_scalar", "VectorField": "fill_halos_vector"}[
            field.__class__.__name__
        ]
    ]
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=NumbaExperimentalFeatureWarning)
        for thread_id in threads:
            sut(thread_id, *meta_and_data, fill_halos_fun, traversals.data.buffer)


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

        # act / plot
        quick_look(advectee, plot)
        fill_halos(advectee, traversals, threads=range(n_threads))
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

        # act / plot
        quick_look(advector, plot)
        fill_halos(advector, traversals, threads=range(n_threads))
        quick_look(advector, plot)

        # assert
        for component in advector.data:
            assert np.isfinite(component).all()

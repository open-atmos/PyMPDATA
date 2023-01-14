# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PyMPDATA import Options, ScalarField, VectorField
from PyMPDATA.boundary_conditions import Extrapolated
from PyMPDATA.impl.traversals import Traversals

JIT_FLAGS = Options().jit_flags


class TestBoundaryConditionExtrapolated:
    @staticmethod
    @pytest.mark.parametrize(
        "data",
        (np.array([1, 2, 3, 4], dtype=float), np.array([1, 2, 3, 4], dtype=complex)),
    )
    def test_1d_scalar(data, n_threads=1, halo=1, left_first=True):
        # arrange
        boundary_conditions = (Extrapolated(),)
        field = ScalarField(data, halo, boundary_conditions)
        # pylint:disable=duplicate-code
        traversals = Traversals(
            grid=field.grid,
            halo=halo,
            jit_flags=JIT_FLAGS,
            n_threads=n_threads,
            left_first=left_first,
        )
        field.assemble(traversals)
        meta_and_data, fill_halos = field.impl
        sut = traversals._code["fill_halos_scalar"]  # pylint:disable=protected-access

        # act
        thread_id = 0
        sut(thread_id, *meta_and_data, *fill_halos)

        # assert
        print(field.data)
        # TODO #289

    @staticmethod
    @pytest.mark.parametrize("data", (np.array([1, 2, 3, 4], dtype=float),))
    def test_1d_vector(data, n_threads=1, halo=2, left_first=True):
        # arrange
        boundary_condition = (Extrapolated(),)
        field = VectorField((data,), halo, boundary_condition)
        # pylint:disable=duplicate-code
        traversals = Traversals(
            grid=field.grid,
            halo=halo,
            jit_flags=JIT_FLAGS,
            n_threads=n_threads,
            left_first=left_first,
        )
        field.assemble(traversals)
        meta_and_data, fill_halos = field.impl
        sut = traversals._code["fill_halos_vector"]  # pylint:disable=protected-access

        # act
        thread_id = 0
        sut(thread_id, *meta_and_data, *fill_halos)

        # assert
        print(field.data)
        # TODO #289

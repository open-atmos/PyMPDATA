# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numba
import numpy as np
import pytest

from PyMPDATA import Options, ScalarField, VectorField
from PyMPDATA.boundary_conditions import Periodic, Polar
from PyMPDATA.impl.enumerations import INNER, OUTER
from PyMPDATA.impl.traversals import Traversals

JIT_FLAGS = Options().jit_flags


class TestPolarBoundaryCondition:
    @staticmethod
    @pytest.mark.parametrize("halo", (1,))
    @pytest.mark.parametrize("n_threads", (1, 2, 3))
    def test_scalar_2d(halo, n_threads):
        # arrange
        data = np.array([[1, 6], [2, 7], [3, 8], [4, 9]], dtype=float)
        boundary_condition = (
            Periodic(),
            Polar(grid=data.shape, longitude_idx=OUTER, latitude_idx=INNER),
        )
        field = ScalarField(data, halo, boundary_condition)
        traversals = Traversals(
            grid=data.shape, halo=halo, jit_flags=JIT_FLAGS, n_threads=n_threads
        )
        field.assemble(traversals)
        meta_and_data, fill_halos = field.impl
        sut = traversals._code["fill_halos_scalar"]  # pylint:disable=protected-access

        # act
        # pylint: disable-next=not-an-iterable
        for thread_id in numba.prange(n_threads):
            sut(thread_id, *meta_and_data, *fill_halos)

        # assert
        np.testing.assert_array_equal(
            field.data[halo:-halo, :halo],
            np.roll(field.get()[:, :halo], data.shape[OUTER] // 2, axis=OUTER),
        )
        np.testing.assert_array_equal(
            field.data[halo:-halo, -halo:],
            np.roll(field.get()[:, -halo:], data.shape[OUTER] // 2, axis=OUTER),
        )

    @staticmethod
    @pytest.mark.parametrize("halo", (1,))
    @pytest.mark.parametrize("n_threads", (1, 2, 3))
    def test_vector_2d(halo, n_threads):
        # arrange
        grid = (4, 2)
        data = (
            np.array(
                [
                    [1, 6],
                    [2, 7],
                    [3, 8],
                    [4, 9],
                    [5, 10],
                ],
                dtype=float,
            ),
            np.array(
                [
                    [1, 5, 9],
                    [2, 6, 10],
                    [3, 7, 11],
                    [4, 8, 12],
                ],
                dtype=float,
            ),
        )
        boundary_conditions = (
            Periodic(),
            Polar(grid=grid, longitude_idx=OUTER, latitude_idx=INNER),
        )
        field = VectorField(data, halo, boundary_conditions)
        traversals = Traversals(
            grid=grid, halo=halo, jit_flags=JIT_FLAGS, n_threads=n_threads
        )
        field.assemble(traversals)
        meta_and_data, fill_halos = field.impl
        sut = traversals._code["fill_halos_vector"]  # pylint:disable=protected-access

        # act
        # pylint: disable-next=not-an-iterable
        for thread_id in numba.prange(n_threads):
            sut(thread_id, *meta_and_data, *fill_halos)

        # assert
        # TODO #228

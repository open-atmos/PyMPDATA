from PyMPDATA import ScalarField, VectorField, \
    PolarBoundaryCondition, PeriodicBoundaryCondition
from PyMPDATA.arakawa_c.traversals import Traversals
from PyMPDATA.arakawa_c.enumerations import OUTER, INNER
import numpy as np
import numba
import pytest


class TestPolarBoundaryCondition:
    @pytest.mark.parametrize("halo", (1, ))
    @pytest.mark.parametrize("n_threads", (1, 2, 3))
    def test_scalar_2d(self, halo, n_threads):
        # arrange
        data = np.array(
            [
                [1,  6],
                [2,  7],
                [3,  8],
                [4,  9]
            ], dtype=float
        )
        bc = (
            PeriodicBoundaryCondition(),
            PolarBoundaryCondition(grid=data.shape, longitude_idx=OUTER, latitude_idx=INNER)
        )
        field = ScalarField(data, halo, bc)
        meta_and_data, fill_halos = field.impl
        traversals = Traversals(grid=data.shape, halo=halo, jit_flags={}, n_threads=n_threads)
        sut = traversals._fill_halos_scalar

        # act
        for thread_id in numba.prange(n_threads):
            sut(thread_id, *meta_and_data, *fill_halos)

        # assert
        np.testing.assert_array_equal(
            field.data[halo:-halo, :halo],
            np.roll(field.get()[:, :halo], data.shape[OUTER] // 2, axis=OUTER)
        )
        np.testing.assert_array_equal(
            field.data[halo:-halo, -halo:],
            np.roll(field.get()[:, -halo:], data.shape[OUTER] // 2, axis=OUTER)
        )

    @pytest.mark.parametrize("halo", (1, ))
    @pytest.mark.parametrize("n_threads", (1, 2, 3))
    def test_vector_2d(self, halo, n_threads):
        # arrange
        grid = (4, 2)
        data = (
            np.array([
                [1,  6],
                [2,  7],
                [3,  8],
                [4,  9],
                [5, 10],
            ], dtype=float),
            np.array([
                [1, 5,  9],
                [2, 6, 10],
                [3, 7, 11],
                [4, 8, 12],
            ], dtype=float)
        )
        bc = (
            PeriodicBoundaryCondition(),
            PolarBoundaryCondition(grid=grid, longitude_idx=OUTER, latitude_idx=INNER)
        )
        field = VectorField(data, halo, bc)
        meta_and_data, fill_halos = field.impl
        traversals = Traversals(grid=grid, halo=halo, jit_flags={}, n_threads=n_threads)
        sut = traversals._fill_halos_vector

        # act
        for thread_id in numba.prange(n_threads):
            sut(thread_id, *meta_and_data, *fill_halos)

        # assert
        # TODO #228

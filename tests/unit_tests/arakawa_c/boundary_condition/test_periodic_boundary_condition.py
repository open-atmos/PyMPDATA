import numpy as np
import pytest
from PyMPDATA import ScalarField, VectorField, PeriodicBoundaryCondition
from PyMPDATA.arakawa_c.traversals import Traversals

# noinspection PyUnresolvedReferences
from ..n_threads_fixture import n_threads


LEFT, RIGHT = 'left', 'right'


def shift(tup, n):
    if not tup or not n:
        return tup
    n %= len(tup)
    return tup[n:] + tup[:n]


def indices(a, b=None, c=None):
    if b is not None:
        if c is not None:
            return (
                slice(a[0] if a[0] else None, a[1] if a[1] else None),
                slice(b[0] if b[0] else None, b[1] if b[1] else None),
                slice(c[0] if c[0] else None, c[1] if c[1] else None)
            )
        return (
            slice(a[0] if a[0] else None, a[1] if a[1] else None),
            slice(b[0] if b[0] else None, b[1] if b[1] else None)
        )
    return slice(a[0], a[1])


class TestPeriodicBoundaryCondition:
    @pytest.mark.parametrize("data", (
            np.array([1, 2, 3]),
            np.array([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]),
            np.arange(3 * 4 * 5).reshape((3, 4, 5))
    ))
    @pytest.mark.parametrize("halo", (1, 2, 3))
    @pytest.mark.parametrize("side", (LEFT, RIGHT))
    def test_scalar(self, data, halo, side, n_threads):
        n_dims = len(data.shape)
        if n_dims == 1 and n_threads > 1:
            return

        # arrange
        field = ScalarField(data, halo, tuple([PeriodicBoundaryCondition() for _ in range(n_dims)]))
        meta_and_data, fill_halos = field.impl
        traversals = Traversals(grid=data.shape, halo=halo, jit_flags={}, n_threads=n_threads)
        sut = traversals._fill_halos_scalar

        # act
        for thread_id in range(n_threads):  # TODO #96: xfail if not all threads executed?
            sut(thread_id, *meta_and_data, *fill_halos)

        # assert
        for dim in range(n_dims):
            if side == LEFT:
                np.testing.assert_array_equal(
                    field.data[shift(indices((None, halo), (halo, -halo), (halo, -halo))[:n_dims], dim)],
                    data[shift(indices((-halo, None), (None, None), (None, None))[:n_dims], dim)]
                )
            else:
                np.testing.assert_array_equal(
                    field.data[shift(indices((-halo, None), (halo, -halo), (halo, -halo))[:n_dims], dim)],
                    data[shift(indices((None, halo), (None, None), (None, None))[:n_dims], dim)]
                )

    @pytest.mark.parametrize("data", (
            (np.array([1, 2, 3]),),
            (
                np.array([
                    [41, 42, 33],
                    [51, 52, 53],
                    [61, 62, 63],
                    [71, 72, 73]
                ]),
                np.array([
                    [11, 12, 13, 14],
                    [21, 22, 23, 24],
                    [31, 32, 33, 34]
                ]),
            )  # TODO #96: 3D
    ))
    @pytest.mark.parametrize("halo", [1, 2, 3])
    @pytest.mark.parametrize("side", [LEFT, RIGHT])
    def test_vector(self, data, halo, side, n_threads):
        n_dims = len(data)
        if n_dims == 1 and n_threads > 1:
            return

        # arrange
        field = VectorField(data, halo, tuple([PeriodicBoundaryCondition() for _ in range(n_dims)]))
        meta_and_data, fill_halos = field.impl
        traversals = Traversals(grid=field.grid, halo=halo, jit_flags={}, n_threads=n_threads)
        sut = traversals._fill_halos_vector

        # act
        for thread_id in range(n_threads):  # TODO #96: xfail if not all threads executed?
            sut(thread_id, *meta_and_data, *fill_halos)

        # assert
        for dim in range(n_dims):
            if n_dims == 1 and halo == 1:
                np.testing.assert_array_equal(field.data[dim], data[dim])
            if side == LEFT:
                np.testing.assert_array_equal(
                    field.data[dim][shift(indices((None, halo - 1), (halo, -halo))[:n_dims], dim)],
                    data[dim][shift(indices((-(halo - 1), None), (None, None))[:n_dims], dim)]
                )
                if n_dims > 1:
                    np.testing.assert_array_equal(
                        field.data[dim][shift(indices((halo-1, -(halo-1)), (None, halo))[:n_dims], dim)],
                        data[dim][shift(indices((None, None), (-halo, None))[:n_dims], dim)]
                    )
            else:
                np.testing.assert_array_equal(
                    field.data[dim][shift(indices((-(halo - 1), None), (halo, -halo))[:n_dims], dim)],
                    data[dim][shift(indices((None, halo - 1), (None, None))[:n_dims], dim)]
                )
                if n_dims > 1:
                    np.testing.assert_array_equal(
                        field.data[dim][shift(indices((halo - 1, -(halo - 1)), (-halo, None))[:n_dims], dim)],
                        data[dim][shift(indices((None, None), (None, halo))[:n_dims], dim)]
                    )

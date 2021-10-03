from PyMPDATA import ScalarField, VectorField, PeriodicBoundaryCondition
from PyMPDATA.arakawa_c.traversals import Traversals
import numpy as np
import pytest
from ..n_threads_fixture import n_threads


LEFT, RIGHT = 'left', 'right'

def shift(tup, n):
    if not tup or not n:
        return tup
    n %= len(tup)
    return tup[n:] + tup[:n]

def indices(a, b, c, d):
    return (
        slice(a if a else None, b if b else None),
        slice(c if c else None, d if d else None)
    )

class TestPeriodicBoundaryCondition:
    @pytest.mark.parametrize("data", (np.array([1, 2, 3]), np.array([[1,2,3],[4,5,6],[7,8,9]])))
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
        for thread_id in range(n_threads):
            sut(thread_id, *meta_and_data, *fill_halos)

        # assert
        if n_dims == 1:
            if side == LEFT:
                np.testing.assert_array_equal(field.data[:halo], data[-halo:])
            else:
                np.testing.assert_array_equal(field.data[-halo:], data[:halo])
        elif n_dims == 2:
            if side == LEFT:
                np.testing.assert_array_equal(field.data[:halo,halo:-halo], data[-halo:,:])
                np.testing.assert_array_equal(field.data[halo:-halo,:halo], data[:,-halo:])
            else:
                np.testing.assert_array_equal(field.data[-halo:,halo:-halo], data[:halo,:])
                np.testing.assert_array_equal(field.data[halo:-halo,-halo:], data[:,:halo])
        else:
            raise NotImplementedError()

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
            )
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
            if n_dims == 1:
                if halo == 1:
                    np.testing.assert_array_equal(field.data[dim], data[dim])
                else:
                    if side == LEFT:
                        np.testing.assert_array_equal(field.data[dim][:(halo-1)], data[dim][-(halo-1):])
                    else:
                        np.testing.assert_array_equal(field.data[dim][-(halo-1):], data[dim][:(halo-1)])
            elif n_dims == 2:
                if side == LEFT:
                    np.testing.assert_array_equal(
                        field.data[dim][shift(indices(None, (halo - 1), halo, -halo), dim)],
                        data[dim][shift(indices(-(halo - 1), None, None, None), dim)]
                    )
                    np.testing.assert_array_equal(
                        field.data[dim][shift(indices((halo-1),-(halo-1), None, halo), dim)],
                        data[dim][shift(indices(None, None, -halo, None), dim)]
                    )
                else:
                    np.testing.assert_array_equal(
                        field.data[dim][shift(indices(-(halo - 1), None, halo, -halo), dim)],
                        data[dim][shift(indices(None, (halo - 1), None, None), dim)]
                    )
                    np.testing.assert_array_equal(
                        field.data[dim][shift(indices((halo - 1), -(halo - 1), -halo, None), dim)],
                        data[dim][shift(indices(None, None, None, halo), dim)]
                    )
        else:
            raise NotImplementedError()

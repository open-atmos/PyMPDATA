from PyMPDATA import ScalarField, VectorField, PeriodicBoundaryCondition
from PyMPDATA.arakawa_c.traversals import Traversals
import numpy as np
import pytest

LEFT, RIGHT = 'left', 'right'


class TestPeriodicBoundaryCondition:
    @pytest.mark.parametrize("data", (np.array([1, 2, 3]),))
    @pytest.mark.parametrize("halo", (1, 2, 3))
    @pytest.mark.parametrize("side", (LEFT, RIGHT))
    def test_scalar_1d(self, data, halo, side):
        # arrange
        field = ScalarField(data, halo, (PeriodicBoundaryCondition(),))
        meta_and_data, fill_halos = field.impl
        traversals = Traversals(grid=data.shape, halo=halo, jit_flags={}, n_threads=1)
        sut = traversals._fill_halos_scalar
        thread_id = 0

        # act
        sut(thread_id, *meta_and_data, *fill_halos)

        # assert
        if side == LEFT:
            np.testing.assert_array_equal(field.data[:halo], data[-halo:])
        elif side == RIGHT:
            np.testing.assert_array_equal(field.data[-halo:], data[:halo])
        else:
            raise ValueError()

    @pytest.mark.parametrize("data", (np.array([1, 2, 3]),))
    @pytest.mark.parametrize("halo", [1, 2, 3])
    @pytest.mark.parametrize("side", [LEFT, RIGHT])
    def test_vector_1d(self, data, halo, side):
        # arrange
        field = VectorField((data,), halo, (PeriodicBoundaryCondition(),))
        meta_and_data, fill_halos = field.impl
        traversals = Traversals(grid=(data.shape[0]-1,), halo=halo, jit_flags={}, n_threads=1)
        sut = traversals._fill_halos_vector
        thread_id = 0

        # act
        sut(thread_id, *meta_and_data, *fill_halos)

        # assert
        if halo == 1:
            return
        if side == LEFT:
            np.testing.assert_array_equal(field.data[0][:(halo-1)], data[-(halo-1):])
        elif side == RIGHT:
            np.testing.assert_array_equal(field.data[0][-(halo-1):], data[:(halo-1)])
        else:
            raise ValueError()

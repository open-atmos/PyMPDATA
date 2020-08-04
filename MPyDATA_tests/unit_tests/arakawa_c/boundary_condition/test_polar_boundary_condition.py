from MPyDATA import ScalarField, PolarBoundaryCondition, PeriodicBoundaryCondition
from MPyDATA.arakawa_c.traversals import Traversals
import numpy as np
import numba
import pytest

LEFT, RIGHT = 'left', 'right'


class TestPeriodicBoundaryCondition:
    @pytest.mark.parametrize("halo", (1, ))
    @pytest.mark.parametrize("side", (LEFT, RIGHT))
    @pytest.mark.parametrize("n_threads", (1,2,3))
    def test_scalar_2d(self, halo, side, n_threads):
        # arrange
        data = np.array(
            [
                [1,  6],
                [2,  7],
                [3,  8],
                [4,  9],
                [5, 10]
            ]
        )
        bc = (
            PeriodicBoundaryCondition(),
            PolarBoundaryCondition(data.shape, 0, 1)
        )
        field = ScalarField(data, halo, bc)
        meta_and_data, fill_halos = field.impl
        traversals = Traversals(grid=data.shape, halo=halo, jit_flags={}, n_threads=n_threads)
        sut = traversals._fill_halos_scalar

        # act
        for thread_id in numba.prange(n_threads):
            sut(thread_id, *meta_and_data, *fill_halos)

        # assert
        print(field.data)
        # TODO

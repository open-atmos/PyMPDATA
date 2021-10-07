import numpy as np
from PyMPDATA import ScalarField
from PyMPDATA.boundary_conditions import Extrapolated
from PyMPDATA.impl.traversals import Traversals
import pytest


class TestBoundaryConditionExtrapolated:
    @pytest.mark.parametrize("data", (
        np.array([1, 2, 3, 4], dtype=float),
        np.array([1, 2, 3, 4], dtype=complex)
    ))
    def test_1d(self, data, n_threads=1, halo=1):
        # arrange
        bc = (Extrapolated(),)
        field = ScalarField(data, halo, bc)
        meta_and_data, fill_halos = field.impl
        traversals = Traversals(grid=data.shape, halo=halo, jit_flags={}, n_threads=n_threads)
        sut = traversals._fill_halos_scalar

        # act
        thread_id = 0
        sut(thread_id, *meta_and_data, *fill_halos)

        # assert
        pass

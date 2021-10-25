# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from PyMPDATA.boundary_conditions import Extrapolated
from PyMPDATA.impl.traversals import Traversals
from PyMPDATA import ScalarField, Options


class TestBoundaryConditionExtrapolated:
    @pytest.mark.parametrize("data", (
        np.array([1, 2, 3, 4], dtype=float),
        np.array([1, 2, 3, 4], dtype=complex)
    ))
    def test_1d_scalar(self, data, n_threads=1, halo=1):
        # arrange
        bc = (Extrapolated(),)
        field = ScalarField(data, halo, bc)
        jit_flags = Options().jit_flags
        traversals = Traversals(grid=data.shape, halo=halo, jit_flags=jit_flags, n_threads=n_threads)
        field.assemble(traversals)
        meta_and_data, fill_halos = field.impl
        sut = traversals._fill_halos_scalar

        # act
        thread_id = 0
        sut(thread_id, *meta_and_data, *fill_halos)

        # assert
        pass

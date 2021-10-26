# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from PyMPDATA.boundary_conditions import Extrapolated
from PyMPDATA.impl.traversals import Traversals
from PyMPDATA import ScalarField, Options


class TestBoundaryConditionExtrapolated:
    @staticmethod
    @pytest.mark.parametrize("data", (
        np.array([1, 2, 3, 4], dtype=float),
        np.array([1, 2, 3, 4], dtype=complex)
    ))
    def test_1d_scalar(data, n_threads=1, halo=1):
        # arrange
        boundary_conditions = (Extrapolated(),)
        field = ScalarField(data, halo, boundary_conditions)
        jit_flags = Options().jit_flags
        traversals = Traversals(
            grid=data.shape, halo=halo, jit_flags=jit_flags, n_threads=n_threads
        )
        field.assemble(traversals)
        meta_and_data, fill_halos = field.impl
        sut = traversals._fill_halos_scalar

        # act
        thread_id = 0
        sut(thread_id, *meta_and_data, *fill_halos)

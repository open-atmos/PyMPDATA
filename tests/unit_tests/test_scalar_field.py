# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
from PyMPDATA import ScalarField
from PyMPDATA.boundary_conditions import Periodic


class TestScalarField:
    @staticmethod
    def test_1d_contiguous():
        grid = (44, )
        data = np.empty(grid)
        boundary_conditions = (Periodic(),)
        sut = ScalarField(data, halo=1, boundary_conditions=boundary_conditions)
        assert sut.get().data.contiguous

    @staticmethod
    def test_2d_first_dim_not_contiguous():
        grid = (44, 44)
        data = np.empty(grid)
        boundary_conditions = (Periodic(), Periodic())
        sut = ScalarField(data, halo=1, boundary_conditions=boundary_conditions)
        assert not sut.get()[:, 0].data.contiguous

    @staticmethod
    def test_2d_second_dim_contiguous():
        grid = (44, 44)
        data = np.empty(grid)
        boundary_conditions = (Periodic(), Periodic())
        sut = ScalarField(data, halo=1, boundary_conditions=boundary_conditions)
        assert sut.get()[0, :].data.contiguous

from PyMPDATA import ScalarField, PeriodicBoundaryCondition
import numpy as np


class TestScalarField:
    @staticmethod
    def test_1d_contiguous():
        grid = (44, )
        data = np.empty(grid)
        bc = (PeriodicBoundaryCondition(),)
        sut = ScalarField(data, halo=1, boundary_conditions=bc)
        assert sut.get().data.contiguous

    @staticmethod
    def test_2d_first_dim_not_contiguous():
        grid = (44, 44)
        data = np.empty(grid)
        bc = (PeriodicBoundaryCondition(), PeriodicBoundaryCondition())
        sut = ScalarField(data, halo=1, boundary_conditions=bc)
        assert not sut.get()[:, 0].data.contiguous

    @staticmethod
    def test_2d_second_dim_contiguous():
        grid = (44, 44)
        data = np.empty(grid)
        bc = (PeriodicBoundaryCondition(), PeriodicBoundaryCondition())
        sut = ScalarField(data, halo=1, boundary_conditions=bc)
        assert sut.get()[0, :].data.contiguous

import MPyDATA.mpi as mpi
from mpi4py.MPI import COMM_WORLD
import pytest


class TestMPI:
    @staticmethod
    def test_init():
        assert mpi.initialized()

    @staticmethod
    @pytest.mark.parametrize("sut", [mpi.size, mpi.size.py_func])
    def test_size(sut):
        size = sut()
        assert size == COMM_WORLD.Get_size()


import MPyDATA.mpi as mpi
from mpi4py.MPI import COMM_WORLD as mpi4py


class TestMPI:
    @staticmethod
    def test_init():
        assert mpi.initialized()

    @staticmethod
    def test_size():
        size = mpi.size()

        assert size == mpi4py.Get_size()


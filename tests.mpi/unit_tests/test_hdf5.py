# pylint: disable=missing-module-docstring,missing-function-docstring,c-extension-no-member

import h5py
import numba
import numba_mpi as mpi
from mpi4py import MPI

from PyMPDATA_MPI.utils import barrier_enclosed


@numba.njit()
def step(dset):
    rank = mpi.rank()
    dset[rank] = rank


def test_hdf5(mpi_tmp_path_fixed):
    path = mpi_tmp_path_fixed / "parallel_test.hdf5"
    rank = mpi.rank()

    with h5py.File(path, "w", driver="mpio", comm=MPI.COMM_WORLD) as file:
        dset = file.create_dataset("test", (mpi.size(),), dtype="i")

        tmp = dset[:]
        step(tmp)
        dset[rank] = tmp[rank]

    with barrier_enclosed():
        with h5py.File(path, "r") as file:
            assert list(file["test"]) == list(range(0, mpi.size()))

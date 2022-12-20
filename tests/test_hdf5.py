# pylint: disable=missing-module-docstring,missing-function-docstring

from pathlib import Path

import h5py
import numba
import numba_mpi as mpi
from mpi4py import MPI

from .fixtures import mpi_tmp_path

assert hasattr(mpi_tmp_path, "_pytestfixturefunction")


RANK = mpi.rank()


@numba.njit()
def step(dset):
    dset[RANK] = RANK


def test_hdf5(mpi_tmp_path):  # pylint: disable=redefined-outer-name
    path = Path(mpi_tmp_path) / "parallel_test.hdf5"

    with h5py.File(path, "w", driver="mpio", comm=MPI.COMM_WORLD) as file:
        dset = file.create_dataset("test", (mpi.size(),), dtype="i")

        tmp = dset[:]
        step(tmp)
        dset[RANK] = tmp[RANK]

    mpi.barrier()

    with h5py.File(path, "r") as file:
        assert list(file["test"]) == list(range(0, mpi.size()))

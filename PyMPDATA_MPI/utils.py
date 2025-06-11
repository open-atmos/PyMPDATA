# pylint: disable=missing-module-docstring,missing-function-docstring

from contextlib import contextmanager

import numba_mpi as mpi
import numpy as np


@contextmanager
def barrier_enclosed():
    try:
        mpi.barrier()
        yield
    finally:
        mpi.barrier()


def setup_dataset_and_sync_all_workers(storage, dataset_name):
    dataset = storage[dataset_name]
    dataset[:] = np.nan
    mpi.barrier()
    return dataset

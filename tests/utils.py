# pylint: disable=missing-module-docstring,missing-function-docstring

from contextlib import contextmanager

import numba_mpi as mpi


@contextmanager
def barrier_enclosed():
    try:
        mpi.barrier()
        yield
    finally:
        mpi.barrier()

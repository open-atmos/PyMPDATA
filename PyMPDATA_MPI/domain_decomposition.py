# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring,invalid-name

import numpy as np
from PyMPDATA.impl.domain_decomposition import make_subdomain
from PyMPDATA.impl.enumerations import OUTER

MPI_DIM = OUTER

subdomain = make_subdomain(jit_flags={})


def mpi_indices(grid, rank, size):
    start, stop = subdomain(grid[MPI_DIM], rank, size)
    xi, yi = np.indices((stop - start, grid[MPI_DIM - 1]), dtype=float)
    xi += start
    return xi, yi

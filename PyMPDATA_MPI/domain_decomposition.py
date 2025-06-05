"""MPI-aware domain decomposition utilities"""

import numpy as np
from PyMPDATA.impl.domain_decomposition import make_subdomain

subdomain = make_subdomain(jit_flags={})


def mpi_indices(*, grid, rank, size, mpi_dim):
    """returns a mapping from rank-local indices to domain-wide indices,
    (subdomain-aware equivalent of np.indices)"""
    start, stop = subdomain(grid[mpi_dim], rank, size)
    indices_arg = list(grid)
    indices_arg[mpi_dim] = stop - start
    xyi = np.indices(tuple(indices_arg), dtype=float)
    xyi[mpi_dim] += start
    return xyi

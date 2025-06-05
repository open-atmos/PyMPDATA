"""
tests for domain decomposition utilities
"""

import pytest
from PyMPDATA.impl.enumerations import INNER, OUTER

from PyMPDATA_MPI.domain_decomposition import mpi_indices


@pytest.mark.parametrize(
    "grid, rank, size, mpi_dim, expected",
    (
        # size=1
        ((2, 2), 0, 1, OUTER, [[[0, 0], [1, 1]], [[0, 1], [0, 1]]]),
        ((2, 2), 0, 1, INNER, [[[0, 0], [1, 1]], [[0, 1], [0, 1]]]),
        # size=2
        ((2, 2), 0, 2, OUTER, [[[0, 0]], [[0, 1]]]),
        ((2, 2), 1, 2, OUTER, [[[1, 1]], [[0, 1]]]),
        ((2, 2), 0, 2, INNER, [[[0], [1]], [[0], [0]]]),
        ((2, 2), 1, 2, INNER, [[[0], [1]], [[1], [1]]]),
    ),
)
def test_mpi_indices(grid, rank, size, mpi_dim, expected):
    """tests the subdomain-aware index-generation logic"""
    # arrange
    sut = mpi_indices

    # act
    xyi = sut(grid=grid, rank=rank, size=size, mpi_dim=mpi_dim)

    # assert
    assert (xyi == expected).all()

"""polar boundary condition logic"""

from functools import lru_cache

import numba
import numba_mpi as mpi
from PyMPDATA.boundary_conditions import Polar
from PyMPDATA.impl.enumerations import INNER, OUTER

from PyMPDATA_MPI.impl import MPIBoundaryCondition


class MPIPolar(MPIBoundaryCondition):
    """class which instances are to be passed in boundary_conditions tuple to the
    `PyMPDATA.scalar_field.ScalarField` and
    `PyMPDATA.vector_field.VectorField` __init__ methods"""

    def __init__(self, mpi_grid, grid, mpi_dim):
        self.worker_pool_size = grid[mpi_dim] // mpi_grid[mpi_dim]
        self.__mpi_size_one = self.worker_pool_size == 1

        if not self.__mpi_size_one:
            only_one_peer_per_subdomain = self.worker_pool_size % 2 == 0
            assert only_one_peer_per_subdomain

        super().__init__(
            size=self.worker_pool_size,
            base=(
                Polar(grid=grid, longitude_idx=OUTER, latitude_idx=INNER)
                if self.__mpi_size_one
                else None
            ),
            mpi_dim=mpi_dim,
        )

    @staticmethod
    def make_vector(indexers, halo, dtype, jit_flags, dimension_index):
        """returns (lru-cached) Numba-compiled vector halo-filling callable"""
        return Polar.make_vector(indexers, halo, dtype, jit_flags, dimension_index)

    @staticmethod
    @lru_cache
    def make_get_peer(jit_flags, size):
        """returns a numba-compiled callable."""

        @numba.njit(**jit_flags)
        def get_peer(_):
            rank = mpi.rank()
            peer = (rank + size // 2) % size
            return peer, peer < size // 2

        return get_peer

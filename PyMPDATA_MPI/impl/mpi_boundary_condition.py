""" common base class for MPI boundary conditions """

from PyMPDATA_MPI.impl.boundary_condition_commons import make_scalar_boundary_condition


class MPIBoundaryCondition:
    """common base class for MPI boundary conditions"""

    def __init__(self, base, size):
        self.__mpi_size_one = size == 1
        self.worker_pool_size = size
        self.base = base

    # pylint: disable=too-many-arguments
    def make_scalar(self, indexers, halo, dtype, jit_flags, dimension_index):
        """returns (lru-cached) Numba-compiled scalar halo-filling callable"""
        if self.__mpi_size_one:
            return self.base.make_scalar(
                indexers, halo, dtype, jit_flags, dimension_index
            )
        return make_scalar_boundary_condition(
            indexers,
            jit_flags,
            dimension_index,
            dtype,
            self.make_get_peer(jit_flags, self.worker_pool_size),
        )

    @staticmethod
    def make_get_peer(_, __):
        """returns (lru-cached) numba-compiled callable."""
        raise NotImplementedError()

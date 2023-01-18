# pylint: disable=invalid-name,unused-argument,c-extension-no-member

""" periodic/cyclic boundary condition logic """
from functools import lru_cache

import numba
import numba_mpi as mpi
import numpy as np
from mpi4py import MPI
from PyMPDATA.boundary_conditions import Periodic
from PyMPDATA.impl.enumerations import SIGN_LEFT, SIGN_RIGHT

TAG = 44
comm = MPI.COMM_WORLD


class MPIPeriodic:
    """class which instances are to be passed in boundary_conditions tuple to the
    `PyMPDATA.scalar_field.ScalarField` and
    `PyMPDATA.vector_field.VectorField` __init__ methods"""

    def __init__(self, size):
        self.__size = size
        assert SIGN_RIGHT == -1
        assert SIGN_LEFT == +1

    def make_scalar(self, ats, _, __, jit_flags):
        """returns (lru-cached) Numba-compiled scalar halo-filling callable"""
        if self.__size == 1:
            return Periodic.make_scalar(ats, _, __, jit_flags)
        return _make_scalar_periodic(ats, jit_flags, self.__size)

    @staticmethod
    def make_vector(ats, _, __, jit_flags):
        """returns (lru-cached) Numba-compiled vector halo-filling callable"""
        return _make_vector_periodic(ats, jit_flags)


@lru_cache()
def _make_scalar_periodic(ats, jit_flags, size):
    @numba.njit(**jit_flags)
    def fill_halos(psi, span, sign):
        rank = mpi.rank()

        peers = (-1, (rank - 1) % size, (rank + 1) % size)  # LEFT  # RIGHT

        buf = np.empty((1,))

        # TODO: take halo size into account when reading data using ats()
        if SIGN_LEFT == sign:
            buf[0] = ats(*psi, sign)
            mpi.send(buf, dest=peers[sign], tag=TAG)
            mpi.recv(buf, source=peers[sign], tag=TAG)
        elif SIGN_RIGHT == sign:
            mpi.recv(buf, source=peers[sign], tag=TAG)
            buf[0] = ats(*psi, sign)
            mpi.send(buf, dest=peers[sign], tag=TAG)

        return buf[0]

    return fill_halos


@lru_cache()
def _make_vector_periodic(ats, jit_flags):
    @numba.njit(**jit_flags)
    def fill_halos(psi, span, sign):
        return ats(*psi, sign * span)

    return fill_halos

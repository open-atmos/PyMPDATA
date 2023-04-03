# pylint: disable=invalid-name,unused-argument,c-extension-no-member,too-many-arguments

""" periodic/cyclic boundary condition logic """
from functools import lru_cache

import numba
import numba_mpi as mpi
import numpy as np
from mpi4py import MPI
from PyMPDATA.boundary_conditions import Periodic
from PyMPDATA.impl.enumerations import INVALID_INDEX, SIGN_LEFT, SIGN_RIGHT

comm = MPI.COMM_WORLD
IRRELEVANT = 666


class MPIPeriodic:
    """class which instances are to be passed in boundary_conditions tuple to the
    `PyMPDATA.scalar_field.ScalarField` and
    `PyMPDATA.vector_field.VectorField` __init__ methods"""

    def __init__(self, size):
        self.__size = size
        assert SIGN_RIGHT == -1
        assert SIGN_LEFT == +1

    def make_scalar(self, indexers, _, __, jit_flags, dimension_index):
        """returns (lru-cached) Numba-compiled scalar halo-filling callable"""
        if self.__size == 1:
            return Periodic.make_scalar(indexers, _, __, jit_flags, dimension_index)
        return _make_scalar_periodic(indexers, jit_flags, dimension_index, self.__size)

    def make_vector(self, indexers, halo, dtype, jit_flags, dimension_index):
        """returns (lru-cached) Numba-compiled vector halo-filling callable"""
        if self.__size == 1:
            return Periodic.make_vector(
                indexers, halo, dtype, jit_flags, dimension_index
            )
        return _make_vector_periodic(
            indexers, halo, jit_flags, dimension_index, self.__size
        )


def _make_send_recv(set_value, jit_flags, fill_buf):
    @numba.njit(**jit_flags)
    def _send_recv(size, psi, i_rng, j_rng, k_rng, sign, dim, output):
        buf = np.empty(
            (
                len(i_rng),
                len(k_rng),
            )
        )

        rank = mpi.rank()
        peers = (-1, (rank - 1) % size, (rank + 1) % size)  # LEFT  # RIGHT

        if SIGN_LEFT == sign:
            fill_buf(buf, psi, i_rng, k_rng, sign, dim)
            mpi.send(buf, dest=peers[sign])
            mpi.recv(buf, source=peers[sign])
        elif SIGN_RIGHT == sign:
            mpi.recv(buf, source=peers[sign])
            tmp = np.empty_like(buf)
            fill_buf(tmp, psi, i_rng, k_rng, sign, dim)
            mpi.send(tmp, dest=peers[sign])

        for i in i_rng:
            for j in j_rng:
                for k in k_rng:
                    set_value(
                        output,
                        i,
                        j,
                        k,
                        buf[i - i_rng.start, k - k_rng.start],
                    )

    return _send_recv


@lru_cache()
def _make_scalar_periodic(indexers, jit_flags, dimension_index, size):
    @numba.njit(**jit_flags)
    def fill_buf(buf, psi, i_rng, k_rng, sign, _dim):
        for i in i_rng:
            for k in k_rng:
                buf[i - i_rng.start, k - k_rng.start] = indexers.ats[dimension_index](
                    (i, INVALID_INDEX, k), psi, sign  # TODO: * halo ?
                )

    send_recv = _make_send_recv(indexers.set, jit_flags, fill_buf)

    @numba.njit(**jit_flags)
    def fill_halos(i_rng, j_rng, k_rng, psi, _, sign):
        send_recv(size, psi, i_rng, j_rng, k_rng, sign, IRRELEVANT, psi)

    return fill_halos


@lru_cache()
def _make_vector_periodic(indexers, halo, jit_flags, dimension_index, size):
    @numba.njit(**jit_flags)
    def fill_buf(buf, components, i_rng, k_rng, sign, dim):
        parallel = dim % len(components) == dimension_index
        assert not parallel

        for i in i_rng:
            for k in k_rng:
                buf[i - i_rng.start, k - k_rng.start] = indexers.atv[dimension_index](
                    (i, INVALID_INDEX, k), components, sign  # TODO: * halo ?
                )

    send_recv = _make_send_recv(indexers.set, jit_flags, fill_buf)

    @numba.njit(**jit_flags)
    def fill_halos_loop_vector(i_rng, j_rng, k_rng, components, dim, _, sign):
        if i_rng.start == i_rng.stop or k_rng.start == k_rng.stop:
            return
        send_recv(size, components, i_rng, j_rng, k_rng, sign, dim, components[dim])

    return fill_halos_loop_vector

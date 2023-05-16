# pylint: disable=invalid-name,unused-argument,c-extension-no-member,too-many-arguments

""" periodic/cyclic boundary condition logic """
from functools import lru_cache

import numba
import numba_mpi as mpi
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

    def make_scalar(self, indexers, halo, dtype, jit_flags, dimension_index):
        """returns (lru-cached) Numba-compiled scalar halo-filling callable"""
        if self.__size == 1:
            return Periodic.make_scalar(
                indexers, halo, dtype, jit_flags, dimension_index
            )
        return _make_scalar_periodic(
            indexers, jit_flags, dimension_index, self.__size, dtype
        )

    def make_vector(self, indexers, halo, dtype, jit_flags, dimension_index):
        """returns (lru-cached) Numba-compiled vector halo-filling callable"""
        if self.__size == 1:
            return Periodic.make_vector(
                indexers, halo, dtype, jit_flags, dimension_index
            )
        return _make_vector_periodic(
            indexers, halo, jit_flags, dimension_index, self.__size, dtype
        )


def _make_send_recv(set_value, jit_flags, fill_buf, size, dtype):
    @numba.njit(**jit_flags)
    def get_buffer_chunk(buffer, i_rng, k_rng, chunk_index):
        chunk_size = len(i_rng) * len(k_rng)
        return buffer.view(dtype)[
            chunk_index * chunk_size : (chunk_index + 1) * chunk_size
        ].reshape((len(i_rng), len(k_rng)))

    @numba.njit(**jit_flags)
    def get_peers():
        rank = mpi.rank()
        left_peer = (rank - 1) % size
        right_peer = (rank + 1) % size
        return (-1, left_peer, right_peer)

    @numba.njit(**jit_flags)
    def fill_output(output, buffer, i_rng, j_rng, k_rng):
        for i in i_rng:
            for j in j_rng:
                for k in k_rng:
                    set_value(
                        output,
                        i,
                        j,
                        k,
                        buffer[i - i_rng.start, k - k_rng.start],
                    )

    @numba.njit(**jit_flags)
    def _send(buf, peer, fill_buf_args):
        fill_buf(buf, *fill_buf_args)
        mpi.send(buf, dest=peer)

    @numba.njit(**jit_flags)
    def _recv(buf, peer):
        mpi.recv(buf, source=peer)

    @numba.njit(**jit_flags)
    def _send_recv(buffer, psi, i_rng, j_rng, k_rng, sign, dim, output):
        buf = get_buffer_chunk(buffer, i_rng, k_rng, chunk_index=0)
        peers = get_peers()
        fill_buf_args = (psi, i_rng, k_rng, sign, dim)

        if SIGN_LEFT == sign:
            _send(buf=buf, peer=peers[sign], fill_buf_args=fill_buf_args)
            _recv(buf=buf, peer=peers[sign])
        elif SIGN_RIGHT == sign:
            _recv(buf=buf, peer=peers[sign])
            tmp = get_buffer_chunk(buffer, i_rng, k_rng, chunk_index=1)
            _send(buf=tmp, peer=peers[sign], fill_buf_args=fill_buf_args)

        fill_output(output, buf, i_rng, j_rng, k_rng)

    return _send_recv


@lru_cache()
def _make_scalar_periodic(indexers, jit_flags, dimension_index, size, dtype):
    @numba.njit(**jit_flags)
    def fill_buf(buf, psi, i_rng, k_rng, sign, _dim):
        for i in i_rng:
            for k in k_rng:
                buf[i - i_rng.start, k - k_rng.start] = indexers.ats[dimension_index](
                    (i, INVALID_INDEX, k), psi, sign
                )

    send_recv = _make_send_recv(indexers.set, jit_flags, fill_buf, size, dtype)

    @numba.njit(**jit_flags)
    def fill_halos(buffer, i_rng, j_rng, k_rng, psi, _, sign):
        send_recv(buffer, psi, i_rng, j_rng, k_rng, sign, IRRELEVANT, psi)

    return fill_halos


@lru_cache()
def _make_vector_periodic(indexers, halo, jit_flags, dimension_index, size, dtype):
    @numba.njit(**jit_flags)
    def fill_buf(buf, components, i_rng, k_rng, sign, dim):
        parallel = dim % len(components) == dimension_index

        for i in i_rng:
            for k in k_rng:
                if parallel:
                    value = indexers.atv[dimension_index](
                        (i, INVALID_INDEX, k), components, sign * halo + 0.5
                    )
                else:
                    value = indexers.atv[dimension_index](
                        (i, INVALID_INDEX, k), components, sign * halo, 0.5
                    )

                buf[i - i_rng.start, k - k_rng.start] = value

    send_recv = _make_send_recv(indexers.set, jit_flags, fill_buf, size, dtype)

    @numba.njit(**jit_flags)
    def fill_halos_loop_vector(buffer, i_rng, j_rng, k_rng, components, dim, _, sign):
        if i_rng.start == i_rng.stop or k_rng.start == k_rng.stop:
            return
        send_recv(buffer, components, i_rng, j_rng, k_rng, sign, dim, components[dim])

    return fill_halos_loop_vector

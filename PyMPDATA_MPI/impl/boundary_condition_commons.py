# pylint: disable=too-many-positional-arguments,too-many-arguments
"""boundary_condition common functions"""

from functools import lru_cache

import numba
import numba_mpi as mpi
from mpi4py import MPI
from PyMPDATA.impl.enumerations import INVALID_INDEX, OUTER

IRRELEVANT = 666


@lru_cache()
def make_scalar_boundary_condition(
    *, indexers, jit_flags, dimension_index, dtype, get_peer, mpi_dim
):
    """returns fill_halos() function for scalar boundary conditions.
    Provides default logic for scalar buffer filling. Notable arguments:
       :param get_peer: function for determining the direction of communication
       :type get_peer: function"""

    @numba.njit(**jit_flags)
    def fill_buf(buf, psi, i_rng, k_rng, sign, _dim):
        for i in i_rng:
            for k in k_rng:
                buf[i - i_rng.start, k - k_rng.start] = indexers.ats[dimension_index](
                    (i, INVALID_INDEX, k), psi, sign
                )

    send_recv = _make_send_recv(
        indexers.set, jit_flags, fill_buf, dtype, get_peer, mpi_dim
    )

    @numba.njit(**jit_flags)
    def fill_halos(buffer, i_rng, j_rng, k_rng, psi, _, sign):
        send_recv(buffer, psi, i_rng, j_rng, k_rng, sign, IRRELEVANT, psi)

    return fill_halos


@lru_cache()
def make_vector_boundary_condition(
    indexers, halo, jit_flags, dimension_index, dtype, get_peer, mpi_dim
):
    """returns fill_halos() function for vector boundary conditions.
    Provides default logic for vector buffer filling. Notable arguments:
       :param get_peer: function for determining the direction of communication
       :type get_peer: function"""

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

    send_recv = _make_send_recv(
        indexers.set, jit_flags, fill_buf, dtype, get_peer, mpi_dim
    )

    @numba.njit(**jit_flags)
    def fill_halos_loop_vector(buffer, i_rng, j_rng, k_rng, components, dim, _, sign):
        if i_rng.start == i_rng.stop or k_rng.start == k_rng.stop:
            return
        send_recv(buffer, components, i_rng, j_rng, k_rng, sign, dim, components[dim])

    return fill_halos_loop_vector


def _make_send_recv(set_value, jit_flags, fill_buf, dtype, get_peer, mpi_dim):

    assert MPI.Query_thread() == MPI.THREAD_MULTIPLE

    @numba.njit(**jit_flags)
    def get_buffer_chunk(buffer, i_rng, k_rng, chunk_index):
        chunk_size = len(i_rng) * len(k_rng)
        if mpi_dim != OUTER:
            n_chunks = len(buffer) // (chunk_size * numba.get_num_threads())
            chunk_index += numba.get_thread_id() * n_chunks
        else:
            n_chunks = len(buffer) // (chunk_size * 2)
            chunk_index += int(numba.get_thread_id() != 0) * n_chunks
        return buffer.view(dtype)[
            chunk_index * chunk_size : (chunk_index + 1) * chunk_size
        ].reshape((len(i_rng), len(k_rng)))

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
        tag = numba.get_thread_id()
        fill_buf(buf, *fill_buf_args)
        mpi.send(buf, dest=peer, tag=tag)

    @numba.njit(**jit_flags)
    def _recv(buf, peer):
        th_id = numba.get_thread_id()
        n_th = numba.get_num_threads()
        tag = th_id if mpi_dim != OUTER else {0: n_th - 1, n_th - 1: 0}[th_id]
        mpi.recv(buf, source=peer, tag=tag)

    @numba.njit(**jit_flags)
    def _send_recv(buffer, psi, i_rng, j_rng, k_rng, sign, dim, output):
        buf = get_buffer_chunk(buffer, i_rng, k_rng, chunk_index=0)
        peer, send_first = get_peer(sign)
        fill_buf_args = (psi, i_rng, k_rng, sign, dim)

        if send_first:
            _send(buf=buf, peer=peer, fill_buf_args=fill_buf_args)
            _recv(buf=buf, peer=peer)
        else:
            _recv(buf=buf, peer=peer)
            tmp = get_buffer_chunk(buffer, i_rng, k_rng, chunk_index=1)
            _send(buf=tmp, peer=peer, fill_buf_args=fill_buf_args)

        fill_output(output, buf, i_rng, j_rng, k_rng)

    return _send_recv

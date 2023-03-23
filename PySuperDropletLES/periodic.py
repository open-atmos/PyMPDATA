# pylint: disable=invalid-name,unused-argument,c-extension-no-member,too-many-arguments

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

    def make_scalar(self, ats, set_value, _, __, jit_flags):
        """returns (lru-cached) Numba-compiled scalar halo-filling callable"""
        if self.__size == 1:
            return Periodic.make_scalar(ats, set_value, _, __, jit_flags)
        return _make_scalar_periodic(ats, set_value, jit_flags, self.__size)

    @staticmethod
    def make_vector(ats, set_value, _, __, jit_flags):
        """returns (lru-cached) Numba-compiled vector halo-filling callable"""
        return _make_vector_periodic(ats, set_value, jit_flags)


@lru_cache()
def _make_scalar_periodic(ats, set_value, jit_flags, size):
    @numba.njit(**jit_flags)
    def _fill_buf(buf, psi, i_rng, j_rng, k_rng, sign):
        for i in i_rng:
            for j in j_rng:
                for k in k_rng:
                    focus = (i, j, k)
                    buf[i - i_rng.start, j - j_rng.start, k - k_rng.start] = ats(
                        focus, psi, sign
                    )

    @numba.njit(**jit_flags)
    def fill_halos(i_rng, j_rng, k_rng, psi, span, sign):
        j_rng = range(j_rng[0], j_rng[0] + 1)
        # addressing
        rank = mpi.rank()
        peers = (-1, (rank - 1) % size, (rank + 1) % size)  # LEFT  # RIGHT

        # allocating (TODO: should not be here!)
        buf = np.empty(
            (
                len(i_rng),
                len(j_rng),
                len(k_rng),
            )
        )
        print(buf.shape)

        # sending/receiving
        if SIGN_LEFT == sign:
            _fill_buf(buf, psi, i_rng, j_rng, k_rng, sign)
            mpi.send(buf, dest=peers[sign], tag=TAG)
            mpi.recv(buf, source=peers[sign], tag=TAG)
        elif SIGN_RIGHT == sign:
            mpi.recv(buf, source=peers[sign], tag=TAG)
            _fill_buf(buf, psi, i_rng, j_rng, k_rng, sign)
            mpi.send(buf, dest=peers[sign], tag=TAG)

        # writing
        for i in i_rng:
            for j in j_rng:
                for k in k_rng:
                    set_value(
                        psi,
                        i,
                        j,
                        k,
                        buf[i - i_rng.start, j - j_rng.start, k - k_rng.start],
                    )

    return fill_halos


@lru_cache()
def _make_vector_periodic(ats, set_value, jit_flags):
    @numba.njit(**jit_flags)
    def fill_halos(psi, span, sign):
        return ats(*psi, sign * span)

    @numba.njit(**jit_flags)
    def fill_halos_loop(i_rng, j_rng, k_rng, psi, span, sign):
        for i in i_rng:
            for j in j_rng:
                for k in k_rng:
                    focus = (i, j, k)
                    set_value(psi, *focus, fill_halos((focus, psi), span, sign))

    return fill_halos_loop

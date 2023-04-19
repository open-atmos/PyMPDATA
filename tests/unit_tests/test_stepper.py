# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from functools import lru_cache

import numba
import numpy as np
import pytest

from PyMPDATA import Options, ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Periodic


def instantiate_solver(*, b_c, buf_size=0):
    n_x = 10
    opt = Options(n_iters=1)
    advector = VectorField(
        data=(np.zeros(n_x + 1),), halo=opt.n_halo, boundary_conditions=b_c
    )
    solver = Solver(
        stepper=Stepper(options=opt, grid=(n_x,), buffer_size=buf_size),
        advectee=ScalarField(
            data=np.zeros(n_x), halo=opt.n_halo, boundary_conditions=b_c
        ),
        advector=advector,
    )
    return solver


class TestStepper:
    @staticmethod
    def test_zero_steps():
        # arrange
        solver = instantiate_solver(b_c=(Periodic(),))

        # act
        time_per_step = solver.advance(0)

        # assert
        assert not np.isfinite(time_per_step)

    @staticmethod
    @pytest.mark.parametrize(
        "buffer_size",
        (
            0,
            1,
            2,
        ),
    )
    def test_buffer(buffer_size):
        # arrange
        VALUE = 44

        class Custom:
            @lru_cache()
            def make_scalar(*args):
                @numba.njit
                def fill_halos(buffer, i_rng, j_rng, k_rng, psi, span, sign):
                    buffer[:] = VALUE

                return fill_halos

            @lru_cache()
            def make_vector(*args):
                @numba.njit
                def fill_halos(buffer, i_rng, j_rng, k_rng, comp, psi, span, sign):
                    buffer[:] = VALUE

                return fill_halos

        solver = instantiate_solver(b_c=(Custom(),), buf_size=buffer_size)

        # act
        solver.advance(1)

        # assert
        buf = solver._Solver__stepper.traversals.data.buffer
        assert (buf == VALUE).all()
        assert buf.size == buffer_size

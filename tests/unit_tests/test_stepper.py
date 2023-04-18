# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import numba
import pytest

from PyMPDATA import Options, ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Periodic


class TestStepper:
    @staticmethod
    def test_zero_steps():
        # arrange
        n_x = 10
        opt = Options(n_iters=1)
        b_c = (Periodic(),)
        advector = VectorField(
            data=(np.zeros(n_x + 1),), halo=opt.n_halo, boundary_conditions=b_c
        )
        solver = Solver(
            stepper=Stepper(options=opt, grid=(n_x,),),
            advectee=ScalarField(
                data=np.zeros(n_x), halo=opt.n_halo, boundary_conditions=b_c
            ),
            advector=advector,
        )

        # act
        time_per_step = solver.advance(0)

        # assert
        assert not np.isfinite(time_per_step)

    @staticmethod
    @pytest.mark.parametrize("buffer_size", (0, 1, 2))
    def test_buffer(buffer_size):
        # arrange
        VALUE = 44

        class Custom:
            def make_scalar(*args):
                @numba.njit
                def fill_halos(buffer, i_rng, j_rng, k_rng, psi, span, sign):
                    assert buffer.shape == (buffer_size,)
                    buffer[:] = VALUE
                return fill_halos

            def make_vector(*args):
                @numba.njit
                def fill_halos(buffer, i_rng, j_rng, k_rng, comp, psi, span, sign):
                    assert buffer.shape == (buffer_size,)
                return fill_halos


        n_x = 10
        opt = Options(n_iters=1)
        b_c = (Custom(),)
        advector = VectorField(
            data=(np.zeros(n_x + 1),), halo=opt.n_halo, boundary_conditions=b_c
        )
        stepper = Stepper(options=opt, grid=(n_x,), buffer_size=buffer_size)
        solver = Solver(
            stepper=stepper,
            advectee=ScalarField(
                data=np.zeros(n_x), halo=opt.n_halo, boundary_conditions=b_c
            ),
            advector=advector,
        )

        # act
        time_per_step = solver.advance(1)

        # assert
        assert (stepper.traversals.data.buffer == VALUE).all()

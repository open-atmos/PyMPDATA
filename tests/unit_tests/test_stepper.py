# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring,too-few-public-methods
import numba
import numpy as np
import pytest

from PyMPDATA import Options, ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Periodic

assert numba.config.DISABLE_JIT is not None  # pylint: disable=no-member


class TestStepper:
    @pytest.mark.skipif("not numba.config.DISABLE_JIT")
    def test_zero_steps():
        # arrange
        n_x = 10
        opt = Options(n_iters=1)
        b_c = (Periodic(),)
        advector = VectorField(
            data=(np.zeros(n_x + 1),), halo=opt.n_halo, boundary_conditions=b_c
        )
        solver = Solver(
            stepper=Stepper(options=opt, grid=(n_x,)),
            advectee=ScalarField(
                data=np.zeros(n_x), halo=opt.n_halo, boundary_conditions=b_c
            ),
            advector=advector,
        )

        # act
        time_per_step = solver.advance(0)

        # assert
        assert not np.isfinite(time_per_step)

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np

from PyMPDATA import Options, ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Periodic


def test_shared_advector():
    n_x = 100
    arr = np.zeros(n_x)
    opt1 = Options(n_iters=2, DPDC=True)
    opt2 = Options(n_iters=2)
    b_c = (Periodic(),)

    halo = opt1.n_halo
    assert opt2.n_halo == halo

    advector = VectorField(
        data=(np.zeros(n_x + 1),), halo=halo, boundary_conditions=b_c
    )
    _ = Solver(
        stepper=Stepper(options=opt1, grid=(n_x,)),
        advectee=ScalarField(data=arr, halo=halo, boundary_conditions=b_c),
        advector=advector,
    )
    solver = Solver(
        stepper=Stepper(options=opt2, grid=(n_x,)),
        advectee=ScalarField(data=arr, halo=halo, boundary_conditions=b_c),
        advector=advector,
    )
    solver.advance(1)

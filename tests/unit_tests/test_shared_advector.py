# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
from PyMPDATA.boundary_conditions import Periodic
from PyMPDATA import ScalarField, VectorField, Solver, Stepper, Options


def test_dtypes():
    nx = 100
    x = np.zeros(nx)
    opt1 = Options(n_iters=2, DPDC=True)
    opt2 = Options(n_iters=2)
    bc = (Periodic(),)

    halo = opt1.n_halo
    assert opt2.n_halo == halo

    advector = VectorField(data=(np.zeros(nx + 1),), halo=halo, boundary_conditions=bc)
    _ = Solver(
        stepper=Stepper(options=opt1, grid=(nx,)),
        advectee=ScalarField(data=x, halo=halo, boundary_conditions=bc),
        advector=advector
    )
    solver = Solver(
        stepper=Stepper(options=opt2, grid=(nx,)),
        advectee=ScalarField(data=x, halo=halo, boundary_conditions=bc),
        advector=advector
    )
    solver.advance(1)

import numpy as np
import pytest
from PyMPDATA import Options, ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Constant, Periodic

grid = 3, 4, 5
courant_numbers = 0.2, 0.3, 0.4  # C=u * dt/dx


@pytest.mark.parametrize(
    "options",
    (
        Options(n_iters=1),
        Options(n_iters=2),
        Options(
            n_iters=3, infinite_gauge=True, nonoscillatory=True, third_order_terms=True
        ),
    ),
)
@pytest.mark.parametrize(
    "stepper_init_args",
    [
        pytest.param({"n_dims": 3}, id="dynamic grid"),
        pytest.param({"grid": grid}, id="static grid"),
    ],
)
def test_pympdata_3d(options, stepper_init_args):
    # arrange
    boundary_conditions = (Periodic(), Periodic(), Constant(0))

    advectees = {
        k: ScalarField(
            np.zeros(grid), halo=options.n_halo, boundary_conditions=boundary_conditions
        )
        for k in ("mass", "energy")
    }
    advector = VectorField(
        tuple(
            np.full(
                tuple(g + (d == dim) for d, g in enumerate(grid)), courant_numbers[dim]
            )
            for dim in range(3)  # 3 components because 3D
        ),
        halo=options.n_halo,
        boundary_conditions=boundary_conditions,
    )
    g_factor = ScalarField(
        np.ones(grid), halo=options.n_halo, boundary_conditions=boundary_conditions
    )

    stepper = Stepper(options=options, **stepper_init_args)
    solvers = {
        k: Solver(stepper, advectee, advector, g_factor)
        for k, advectee in advectees.items()
    }

    for solver in solvers.values():
        solver.advance(n_steps=1)

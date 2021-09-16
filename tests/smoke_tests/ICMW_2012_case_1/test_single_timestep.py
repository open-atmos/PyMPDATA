from PyMPDATA import Stepper, ScalarField, PeriodicBoundaryCondition, Solver, Options, VectorField
from PyMPDATA_examples.utils import nondivergent_vector_field_2d
import numpy as np
import pytest


@pytest.mark.parametrize(
    "options", [
        Options(n_iters=1),
        Options(n_iters=2),
        Options(n_iters=2, flux_corrected_transport=True),
        Options(n_iters=3, flux_corrected_transport=True),
        Options(n_iters=2, flux_corrected_transport=True, infinite_gauge=True),
        Options(flux_corrected_transport=True, infinite_gauge=True, third_order_terms=True),
        Options(flux_corrected_transport=False, infinite_gauge=True),
        Options(flux_corrected_transport=False, third_order_terms=True),
        Options(flux_corrected_transport=False, infinite_gauge=True, third_order_terms=True)
    ]
)
def test_single_timestep(options):
    # Arrange
    grid = (75, 75)
    size = (1500, 1500)
    dt = 1
    w_max = .6

    def stream_function(xX, zZ):
        X = size[0]
        return - w_max * X / np.pi * np.sin(np.pi * zZ) * np.cos(2 * np.pi * xX)

    rhod_of = lambda z: 1 - z * 1e-4
    rhod = np.repeat(
        rhod_of(
            (np.arange(grid[1]) + 1 / 2) / grid[1]
        ).reshape((1, grid[1])),
        grid[0],
        axis=0
    )

    values = {'th': np.full(grid, 300), 'qv': np.full(grid, .001)}
    stepper = Stepper(options=options, grid=grid, non_unit_g_factor=True)
    advector = nondivergent_vector_field_2d(grid, size, dt, stream_function, options.n_halo)
    g_factor = ScalarField(rhod.astype(dtype=options.dtype), halo=options.n_halo,
                           boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition()))
    mpdatas = {}
    for k1, v1 in values.items():
        advectee = ScalarField(np.full(grid, v1, dtype=options.dtype), halo=options.n_halo,
                               boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition()))
        mpdatas[k1] = Solver(stepper=stepper, advectee=advectee, advector=advector, g_factor=g_factor)

    # Act
    for mpdata in mpdatas.values():
        mpdata.advance(nt=1)

    # Assert
    for k, v in mpdatas.items():
        assert np.isfinite(v.advectee.get()).all()

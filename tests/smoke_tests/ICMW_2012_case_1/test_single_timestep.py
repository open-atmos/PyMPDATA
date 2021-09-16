from PyMPDATA import Stepper, ScalarField, PeriodicBoundaryCondition, Solver, Options, VectorField
import numpy as np
import pytest


def nondivergent_vector_field_2d(grid, size, dt, stream_function: callable, halo):
    dx = size[0] / grid[0]
    dz = size[1] / grid[1]
    dxX = 1 / grid[0]
    dzZ = 1 / grid[1]

    xX, zZ = x_vec_coord(grid)
    rho_velocity_x = -(stream_function(xX, zZ + dzZ/2) - stream_function(xX, zZ - dzZ/2)) / dz

    xX, zZ = z_vec_coord(grid)
    rho_velocity_z = (stream_function(xX + dxX/2, zZ) - stream_function(xX - dxX/2, zZ)) / dx

    GC = [rho_velocity_x * dt / dx, rho_velocity_z * dt / dz]

    # CFL condition
    for d in range(len(GC)):
        np.testing.assert_array_less(np.abs(GC[d]), 1)

    result = VectorField(GC, halo=halo, boundary_conditions=(PeriodicBoundaryCondition(), PeriodicBoundaryCondition()))

    # nondivergence (of velocity field, hence dt)
    assert np.amax(abs(result.div((dt, dt)).get())) < 5e-9

    return result


def x_vec_coord(grid):
    nx = grid[0]+1
    nz = grid[1]
    xX = np.repeat(np.linspace(0, grid[0], nx).reshape((nx, 1)), nz, axis=1) / grid[0]
    assert np.amin(xX) == 0
    assert np.amax(xX) == 1
    assert xX.shape == (nx, nz)
    zZ = np.repeat(np.linspace(1 / 2, grid[1] - 1/2, nz).reshape((1, nz)), nx, axis=0) / grid[1]
    assert np.amin(zZ) >= 0
    assert np.amax(zZ) <= 1
    assert zZ.shape == (nx, nz)
    return xX, zZ


def z_vec_coord(grid):
    nx = grid[0]
    nz = grid[1]+1
    xX = np.repeat(np.linspace(1/2, grid[0]-1/2, nx).reshape((nx, 1)), nz, axis=1) / grid[0]
    assert np.amin(xX) >= 0
    assert np.amax(xX) <= 1
    assert xX.shape == (nx, nz)
    zZ = np.repeat(np.linspace(0, grid[1], nz).reshape((1, nz)), nx, axis=0) / grid[1]
    assert np.amin(zZ) == 0
    assert np.amax(zZ) == 1
    assert zZ.shape == (nx, nz)
    return xX, zZ


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

import numpy as np

from PyMPDATA import VectorField
from PyMPDATA.boundary_conditions import Periodic


def nondivergent_vector_field_2d(grid, size, dt, stream_function: callable, halo):
    dx = size[0] / grid[0]
    dz = size[1] / grid[1]
    dxX = 1 / grid[0]
    dzZ = 1 / grid[1]

    xX, zZ = x_vec_coord(grid)
    rho_velocity_x = (
        -(stream_function(xX, zZ + dzZ / 2) - stream_function(xX, zZ - dzZ / 2)) / dz
    )

    xX, zZ = z_vec_coord(grid)
    rho_velocity_z = (
        stream_function(xX + dxX / 2, zZ) - stream_function(xX - dxX / 2, zZ)
    ) / dx

    GC = [rho_velocity_x * dt / dx, rho_velocity_z * dt / dz]

    # CFL condition
    for val in GC:
        np.testing.assert_array_less(np.abs(val), 1)

    result = VectorField(GC, halo=halo, boundary_conditions=(Periodic(), Periodic()))

    # nondivergence (of velocity field, hence dt)
    assert np.amax(abs(result.div((dt, dt)).get())) < 5e-9

    return result


def x_vec_coord(grid):
    nx = grid[0] + 1
    nz = grid[1]
    xX = np.repeat(np.linspace(0, grid[0], nx).reshape((nx, 1)), nz, axis=1) / grid[0]
    assert np.amin(xX) == 0
    assert np.amax(xX) == 1
    assert xX.shape == (nx, nz)
    zZ = (
        np.repeat(np.linspace(1 / 2, grid[1] - 1 / 2, nz).reshape((1, nz)), nx, axis=0)
        / grid[1]
    )
    assert np.amin(zZ) >= 0
    assert np.amax(zZ) <= 1
    assert zZ.shape == (nx, nz)
    return xX, zZ


def z_vec_coord(grid):
    nx = grid[0]
    nz = grid[1] + 1
    xX = (
        np.repeat(np.linspace(1 / 2, grid[0] - 1 / 2, nx).reshape((nx, 1)), nz, axis=1)
        / grid[0]
    )
    assert np.amin(xX) >= 0
    assert np.amax(xX) <= 1
    assert xX.shape == (nx, nz)
    zZ = np.repeat(np.linspace(0, grid[1], nz).reshape((1, nz)), nx, axis=0) / grid[1]
    assert np.amin(zZ) == 0
    assert np.amax(zZ) == 1
    assert zZ.shape == (nx, nz)
    return xX, zZ

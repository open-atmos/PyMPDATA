"""
Created at 21.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np
from .arakawa_c.scalar_field import ScalarField
from .arakawa_c.vector_field import VectorField
from .mpdata import MPDATA
from .options import Options
from .eulerian_fields import EulerianFields


class MPDATAFactory:
    @staticmethod
    def n_halo(opts: Options):
        if opts.n_iters > 1 and (opts.dfl or opts.fct or opts.tot):
            n_halo = 2
        else:
            n_halo = 1
        return n_halo

    @staticmethod
    def uniform_C_1d(psi: np.ndarray, C: float, opts: Options):
        nx = psi.shape[0]
        halo = MPDATAFactory.n_halo(opts)

        state = ScalarField(psi, halo)
        GC = VectorField(data=[np.full((nx + 1,), C)], halo=halo)
        g_factor = ScalarField(np.ones((nx,)), halo=0)  # TODO
        return MPDATAFactory._mpdata(state=state, GC_field=GC, g_factor=g_factor, opts=opts)

    @staticmethod
    def uniform_C_2d(psi: np.ndarray, C: iter, opts: Options):
        nx = psi.shape[0]
        ny = psi.shape[1]
        halo = MPDATAFactory.n_halo(opts)

        state = ScalarField(psi, halo)
        GC = VectorField(data=[
            np.full((nx + 1, ny), C[0]),
            np.full((nx, ny+1), C[1])
        ], halo=halo)
        g_factor = ScalarField(np.ones((nx,ny)), halo=0)  # TODO
        return MPDATAFactory._mpdata(state=state, GC_field=GC, g_factor=g_factor, opts=opts)

    @staticmethod
    def kinematic_2d(grid, size, dt, stream_function: callable, field_values: dict, g_factor: np.ndarray, opts):
        halo = MPDATAFactory.n_halo(opts)
        GC = _nondivergent_vector_field_2d(grid, size, halo, dt, stream_function)
        G = ScalarField(g_factor, halo=0)

        mpdatas = {}
        for key, value in field_values.items():
            state = _uniform_scalar_field(grid, value, halo)
            mpdatas[key] = MPDATAFactory._mpdata(state=state, GC_field=GC, g_factor=G, opts=opts)

        eulerian_fields = EulerianFields(mpdatas)
        return GC, eulerian_fields

    @staticmethod
    def _mpdata(
            state: ScalarField,
            g_factor: ScalarField,
            GC_field: VectorField,
            opts: Options
    ):
        # TODO: move to tests
        if len(state.shape) == 2:
            assert state._impl._data.shape[0] == GC_field._impl._data_0.shape[0] + 1
            assert state._impl._data.shape[1] == GC_field._impl._data_0.shape[1]
            assert GC_field._impl._data_0.shape[0] == GC_field._impl._data_1.shape[0] - 1
            assert GC_field._impl._data_0.shape[1] == GC_field._impl._data_1.shape[1] + 1
        # TODO: assert G.data.shape == state.data.shape (but halo...)
        # TODO assert halo

        prev = ScalarField.full_like(state)  # TODO rename?
        GC_antidiff = VectorField.full_like(GC_field)
        flux = VectorField.full_like(GC_field)
        halo = state.halo

        if (opts.n_iters != 1) & opts.fct:
            psi_min = ScalarField.full_like(state)
            psi_max = ScalarField.full_like(state)
            beta_up = ScalarField.full_like(state)
            beta_dn = ScalarField.full_like(state)
        else:
            psi_min = None
            psi_max = None
            beta_up = None
            beta_dn = None

        mpdata = MPDATA(curr=state, prev=prev, G=g_factor, GC_physical=GC_field, GC_antidiff=GC_antidiff, flux=flux,
                        psi_min=psi_min, psi_max=psi_max, beta_up=beta_up, beta_dn=beta_dn,
                        opts=opts, halo=halo)

        return mpdata


def _uniform_scalar_field(grid, value: float, halo: int):
    data = np.full(grid, value)
    return ScalarField(data=data, halo=halo)


# TODO: move asserts to a unit test
def x_vec_coord(grid, size):
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


# TODO: move asserts to a unit test
def z_vec_coord(grid, size):
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


def _nondivergent_vector_field_2d(grid, size, halo, dt, stream_function: callable):
    # TODO: density!
    dx = size[0] / grid[0]
    dz = size[1] / grid[1]
    dxX = 1 / grid[0]
    dzZ = 1 / grid[1]

    xX, zZ = x_vec_coord(grid, size)
    rho_velocity_x = -(stream_function(xX, zZ + dzZ/2) - stream_function(xX, zZ - dzZ/2)) / dz

    xX, zZ = z_vec_coord(grid, size)
    rho_velocity_z = (stream_function(xX + dxX/2, zZ) - stream_function(xX - dxX/2, zZ)) / dx

    GC = [rho_velocity_x * dt / dx, rho_velocity_z * dt / dz]

    # CFL condition
    for d in range(len(GC)):
        np.testing.assert_array_less(np.abs(GC[d]), 1)

    result = VectorField(data=GC, halo=halo)

    # nondivergence (of velocity field, hence dt)
    assert np.amax(abs(result.div((dt, dt)).get())) < 5e-9

    return result

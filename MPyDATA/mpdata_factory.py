"""
Created at 21.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np
from .arakawa_c.scalar_field import ScalarField
from .arakawa_c.vector_field import VectorField
from .arakawa_c.boundary_conditions.cyclic import CyclicLeft, CyclicRight
from .arakawa_c.boundary_conditions.extrapolated import ExtrapolatedLeft, ExtrapolatedRight
from .mpdata import MPDATA
from .options import Options
from .eulerian_fields import EulerianFields


class MPDATAFactory:
    @staticmethod
    def n_halo(opts: Options):
        if opts.dfl or opts.fct or opts.tot:
            n_halo = 2
        else:
            n_halo = 1
        return n_halo

    @staticmethod
    def uniform_C_1d(psi: np.ndarray, C: float, opts: Options, boundary_conditions):
        nx = psi.shape[0]
        halo = MPDATAFactory.n_halo(opts)

        state = ScalarField(psi, halo, boundary_conditions=boundary_conditions)
        GC = VectorField(data=[np.full((nx + 1,), C)], halo=halo, boundary_conditions=boundary_conditions)
        g_factor = ScalarField(np.ones((nx,)), halo=halo, boundary_conditions=boundary_conditions)  # TODO: nug:False?
        return MPDATA(state=state, GC_field=GC, g_factor=g_factor, opts=opts)

    @staticmethod
    def equilibrium_growth_C_1d(nr, r_min, r_max, dt, coord, cdf, drdt, opts: Options):
        # TODO
        assert opts.nug

        _, dx = np.linspace(
            coord.x(r_min),
            coord.x(r_max),
            nr + 1,
            retstep=True
        )
        xh = np.linspace(
            coord.x(r_min),
            coord.x(r_max),
            nr + 1
        )
        rh = coord.r(xh)
        Gh = 1 / coord.dx_dr(rh)
        x = np.linspace(
            xh[0] + dx / 2,
            xh[-1] - dx / 2,
            nr
        )
        r = coord.r(x)
        G = 1 / coord.dx_dr(r)

        psi = (
            np.diff(cdf(rh))
            /
            np.diff(rh)
        )

        # C = drdt * dxdr * dt / dx
        # G = 1 / dxdr
        C = drdt(rh) / Gh * dt / dx
        GCh = Gh * C

        bcond = ((ExtrapolatedLeft, ExtrapolatedRight),)
        n_halo = MPDATAFactory.n_halo(opts)
        g_factor = ScalarField(G, halo=n_halo, boundary_conditions=bcond)
        state = ScalarField(psi, halo=n_halo, boundary_conditions=bcond)
        GC_field = VectorField([GCh], halo=n_halo, boundary_conditions=bcond)
        return (
            MPDATA(g_factor=g_factor, opts=opts, state=state, GC_field=GC_field),
            r[n_halo:-n_halo],
            rh[(n_halo-1):nr+1-(n_halo-1)]
        )


    @staticmethod
    def uniform_C_2d(psi: np.ndarray, C: iter, opts: Options):
        # TODO
        bcond = (
            (CyclicLeft(), CyclicRight()),
            (CyclicLeft(), CyclicRight())
        )

        nx = psi.shape[0]
        ny = psi.shape[1]
        halo = MPDATAFactory.n_halo(opts)

        state = ScalarField(psi, halo, boundary_conditions=bcond)
        GC = VectorField(data=[
            np.full((nx + 1, ny), C[0]),
            np.full((nx, ny+1), C[1])
        ], halo=halo, boundary_conditions=bcond)
        g_factor = ScalarField(np.ones((nx, ny)), halo=halo, boundary_conditions=bcond)  # TODO
        return MPDATA(state=state, GC_field=GC, g_factor=g_factor, opts=opts)

    @staticmethod
    def kinematic_2d(grid, size, dt, stream_function: callable, field_values: dict, g_factor: np.ndarray, opts):
        # TODO
        bcond = (
            (CyclicLeft(), CyclicRight()),
            (CyclicLeft(), CyclicRight())
        )

        halo = MPDATAFactory.n_halo(opts)
        GC = _nondivergent_vector_field_2d(grid, size, halo, dt, stream_function, boundary_conditions=bcond)
        G = ScalarField(g_factor, halo=halo, boundary_conditions=bcond)

        mpdatas = {}
        for key, value in field_values.items():
            state = _uniform_scalar_field(grid, value, halo, boundary_conditions=bcond)
            mpdatas[key] = MPDATA(opts=opts, state=state, GC_field=GC, g_factor=G)

        eulerian_fields = EulerianFields(mpdatas)
        return GC, eulerian_fields


def _uniform_scalar_field(grid, value: float, halo: int, boundary_conditions):
    data = np.full(grid, value)
    return ScalarField(data=data, halo=halo, boundary_conditions=boundary_conditions)


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


def _nondivergent_vector_field_2d(grid, size, halo, dt, stream_function: callable, boundary_conditions):
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

    result = VectorField(data=GC, halo=halo, boundary_conditions=boundary_conditions)

    # nondivergence (of velocity field, hence dt)
    assert np.amax(abs(result.div((dt, dt)).get())) < 5e-9

    return result

"""
Created at 21.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np
from .arakawa_c.scalar_field import ScalarField
from .arakawa_c.vector_field import VectorField
from .arakawa_c.scalar_constant import ScalarConstant
from .arakawa_c.boundary_conditions.cyclic import CyclicLeft, CyclicRight
from .arakawa_c.boundary_conditions.extrapolated import ExtrapolatedLeft, ExtrapolatedRight
from .arakawa_c.boundary_conditions.zero import ZeroLeft, ZeroRight
from .mpdata import MPDATA
from .options import Options
from .eulerian_fields import EulerianFields
from scipy import integrate
from .utils.pdf_integrator import discretised_analytical_solution


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
        g_factor = ScalarConstant(1)
        return MPDATA(state=state, GC_field=GC, g_factor=g_factor, opts=opts)

    @staticmethod
    def equilibrium_growth_C_1d(nr, r_min, r_max, dt, grid_coord, psi_coord, pdf, drdt, opts: Options):
        # TODO !!!!!!!!!!!!!!!!!!!!!!1
        # if not isinstance(grid_coord, x_id):
        #     assert opts.nug
        # else:
        #     assert not opts.nug

        xh, dx = np.linspace(
            grid_coord.x(r_min),
            grid_coord.x(r_max),
            nr + 1,
            retstep=True
        )
        rh = grid_coord.r(xh)

        x = np.linspace(
            xh[0] + dx / 2,
            xh[-1] - dx / 2,
            nr
        )
        r = grid_coord.r(x)
        G = 1 / grid_coord.dx_dr(r)
        psi = discretised_analytical_solution(rh, lambda r: psi_coord.from_n_n(pdf(r), r))

        # C = drdt * dxdr * dt / dx
        # G = 1 / dxdr
        GCh = psi_coord.dx_dr(rh) * drdt(rh) * dt / dx

        # CFL condition
        np.testing.assert_array_less(np.abs(GCh), 1)

        bcond_extrapol = ((ExtrapolatedLeft, ExtrapolatedRight),)
        bcond_zero = ((ZeroLeft, ZeroRight),)
        n_halo = MPDATAFactory.n_halo(opts)
        g_factor = ScalarField(G, halo=n_halo, boundary_conditions=bcond_extrapol)
        state = ScalarField(psi, halo=n_halo, boundary_conditions=bcond_zero)
        GC_field = VectorField([GCh], halo=n_halo, boundary_conditions=bcond_zero)
        return (
            MPDATA(g_factor=g_factor, opts=opts, state=state, GC_field=GC_field),
            r,
            rh,
            dx
        )

    # TODO: only used in tests -> move to tests
    @staticmethod
    def uniform_C_2d(psi: np.ndarray, C: iter, opts: Options):
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
        g_factor = ScalarField(np.ones((nx, ny)), halo=halo, boundary_conditions=bcond)
        return MPDATA(state=state, GC_field=GC, g_factor=g_factor, opts=opts)

    @staticmethod
    def kinematic_2d(grid, size, dt, stream_function: callable, field_values: dict, opts: Options,
                     g_factor: [np.ndarray, None] = None):
        # TODO
        bcond = (
            (CyclicLeft(), CyclicRight()),
            (CyclicLeft(), CyclicRight())
        )

        halo = MPDATAFactory.n_halo(opts)
        GC = _nondivergent_vector_field_2d(grid, size, halo, dt, stream_function, boundary_conditions=bcond)

        if g_factor is not None:
            assert opts.nug
            G = ScalarField(g_factor, halo=halo, boundary_conditions=bcond)
        else:
            assert not opts.nug
            G = ScalarConstant(1)

        mpdatas = {}
        for key, data in field_values.items():
            state = ScalarField(data=data, halo=halo, boundary_conditions=bcond)
            mpdatas[key] = MPDATA(opts=opts, state=state, GC_field=GC, g_factor=G)

        eulerian_fields = EulerianFields(mpdatas)
        return GC, eulerian_fields

    @staticmethod
    def from_cdf_1d(cdf: callable, x_min: float, x_max: float, nx: int):
        dx = (x_max - x_min) / nx
        x = np.linspace(x_min + dx / 2, x_max - dx / 2, nx)
        xh = np.linspace(x_min, x_max, nx + 1)
        y = np.diff(cdf(xh)) / dx
        return x, y


    @staticmethod
    def from_pdf_2d(pdf: callable, xrange: list, yrange: list, gridsize: list):
        z = np.empty(gridsize)
        dx, dy = (xrange[1] - xrange[0]) / gridsize[0], (yrange[1] - yrange[0]) / gridsize[1]
        for i in range(gridsize[0]):
            for j in range(gridsize[1]):
                z[i, j] = integrate.nquad(pdf, ranges=(
                    (xrange[0] + dx*i, xrange[0] + dx*(i+1)),
                    (yrange[0] + dy*j, yrange[0] + dy*(j+1))
                ))[0] / dx / dy
        x = np.linspace(xrange[0] + dx / 2, xrange[1] - dx / 2, gridsize[0])
        y = np.linspace(yrange[0] + dy / 2, yrange[1] - dy / 2, gridsize[1])
        return x, y, z

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




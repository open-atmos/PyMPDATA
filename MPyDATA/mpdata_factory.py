"""
Created at 21.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""
import numpy as np
import numba
from scipy import integrate

from .arakawa_c.vector_field import VectorField
from .arakawa_c.scalar_field import ScalarField
from .eulerian_fields import EulerianFields
from .mpdata import MPDATA
from .formulae.jit_flags import jit_flags
from .formulae.upwind import make_upwind
from .formulae.flux import make_flux
from .arakawa_c.utils import at_1d, at_2d, atv_1d, atv_2d, set_2d, set_1d, get_2d, get_1d


class MPDATAFactory:
    @staticmethod
    def constant_1d(data, C):
        halo = 1  # TODO

        mpdata = MPDATA(
            step_impl=make_step(data.shape, halo=halo, non_unit_g_factor=False),
            advectee=ScalarField(data, halo=halo),
            advector=VectorField((np.full(data.shape[0] + 1, C),), halo=halo)
        )
        return mpdata

    @staticmethod
    def constant_2d(data, C):
        halo = 1 # TODO
        grid = data.shape
        GC_data = [
            np.full((grid[0] + 1, grid[1]), C[0]),
            np.full((grid[0], grid[1] + 1), C[1])
        ]
        GC = VectorField(GC_data, halo=halo)
        state = ScalarField(data=data, halo=halo)
        step = make_step(grid, halo, non_unit_g_factor=False)
        mpdata = MPDATA(step_impl=step, advectee=state, advector=GC)
        return mpdata

    @staticmethod
    def stream_function_2d_basic(grid, size, dt, stream_function, field):
        halo = 1 # TODO
        step = make_step(grid, halo, non_unit_g_factor=False)
        GC = nondivergent_vector_field_2d(grid, size, dt, stream_function, halo)
        advectee = ScalarField(field, halo=halo)
        return MPDATA(step, advectee=advectee, advector=GC)

    @staticmethod
    def stream_function_2d(grid, size, dt, stream_function, field_values, g_factor):
        halo = 1 # TODO
        step = make_step(grid, halo, non_unit_g_factor=True)
        GC = nondivergent_vector_field_2d(grid, size, dt, stream_function, halo)
        g_factor = ScalarField(g_factor, halo=halo)
        mpdatas = {}
        for k, v in field_values.items():
            advectee = ScalarField(np.full(grid, v), halo=halo)
            mpdatas[k] = MPDATA(step, advectee=advectee, advector=GC, g_factor=g_factor)
        return GC, EulerianFields(mpdatas)



# TODO: new file


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

    result = VectorField(GC, halo=halo)

    # nondivergence (of velocity field, hence dt)
    assert np.amax(abs(result.div((dt, dt)).get())) < 5e-9

    return result


# TODO: move asserts to a unit test
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


# TODO: move asserts to a unit test
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


def make_step(grid, halo, non_unit_g_factor):


    n_dims = len(grid)
    ni = grid[0]
    nj = grid[1] if n_dims > 1 else 0


    if n_dims == 1:
        at = at_1d
        atv = atv_1d
        set = set_1d
        get = get_1d
    elif n_dims == 2:
        at = at_2d
        atv = atv_2d
        set = set_2d
        get = get_2d
    else:
        raise NotImplementedError

    @numba.njit(**jit_flags)
    def apply_vector(fun, out_0, out_1, prev, GC_phys_0, GC_phys_1):
        GC_phys_tpl = (GC_phys_0, GC_phys_1)
        out_tpl = (out_0, out_1)
        # -1, -1
        for i in range(halo-1, ni+1+halo-1):
            for j in range(halo-1, nj+1+halo-1) if n_dims > 1 else [-1]:
                focus = (0, i, j)
                set(out_tpl[0], i, j, fun(focus, prev, GC_phys_tpl))
                if n_dims > 1:
                    focus = (1, i, j)
                    set(out_tpl[1], i, j, fun(focus, prev, GC_phys_tpl))

    @numba.njit(**jit_flags)
    def apply_scalar(fun, out,
                     flux_0, flux_1,
                     g_factor
                     ):
        flux_tpl = (flux_0, flux_1)
        for i in range(halo, ni+halo):
            for j in range(halo, nj+halo) if n_dims > 1 else [-1]:
                focus = (0, i, j)
                set(out, i, j, fun(focus, get(out, i, j), flux_tpl, g_factor))
                if n_dims > 1:
                    focus = (1, i, j)
                    set(out, i, j, fun(focus, get(out, i, j), flux_tpl, g_factor))

    flux = make_flux(atv, at)
    upwind = make_upwind(atv, at, non_unit_g_factor)

    @numba.njit(**jit_flags)
    def boundary_cond(psi, fun):
        # TODO: d-dimensions
        # psi[0:halo, :] = psi[-2 * halo:-halo, :]
        # psi[:, 0:halo] = psi[:, -2 * halo:-halo]
        # psi[-halo:, :] = psi[halo:2 * halo, :]
        # psi[:, -halo:] = psi[:, halo:2 * halo]
        for i in range(0, halo):
            for j in range(0, nj+2*halo) if n_dims > 1 else [-1]:
                focus = (0, i, j)
                set(psi, i, j, fun(focus, psi, ni, 1))
        for i in range(ni+halo, ni+2*halo):
            for j in range(0, nj+2*halo) if n_dims > 1 else [-1]:
                focus = (0, i, j)
                set(psi, i, j, fun(focus, psi, ni, -1))
        if n_dims > 1:
            for j in range(0, halo):
                for i in range(0, ni + 2 * halo):
                    focus = (1, i, j)
                    set(psi, i, j, fun(focus, psi, nj, 1))
            for j in range(nj+halo, nj+2*halo):
                for i in range(0, ni + 2 * halo):
                    focus = (1, i, j)
                    set(psi, i, j, fun(focus, psi, nj, -1))
        pass

    @numba.njit(**jit_flags)
    def cyclic(focus, psi, n, sign):
        return at(focus, psi, sign*n, 0)

    @numba.njit(**jit_flags)
    def step(nt, psi, flux_0, flux_1, GC_phys_0, GC_phys_1, g_factor):
        flux_tpl = (flux_0, flux_1)
        GC_phys = (GC_phys_0, GC_phys_1)

        for _ in range(nt):
            boundary_cond(psi, cyclic)
            apply_vector(flux, *flux_tpl, psi, *GC_phys)
            apply_scalar(upwind, psi, *flux_tpl, g_factor)
    return step


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
from .formulae.flux import make_flux_first_pass, make_flux_subsequent
from .formulae.antidiff import make_antidiff
from .options import Options
from .arakawa_c.utils import at_1d, at_2d, atv_1d, atv_2d_0, atv_2d_1, set_2d, set_1d, get_2d, get_1d


class MPDATAFactory:
    @staticmethod
    def constant_1d(data, C, options: Options):
        halo = 1  # TODO

        mpdata = MPDATA(
            step_impl=make_step(data.shape, halo=halo, non_unit_g_factor=False, options=options),
            advectee=ScalarField(data, halo=halo),
            advector=VectorField((np.full(data.shape[0] + 1, C),), halo=halo)
        )
        return mpdata

    @staticmethod
    def constant_2d(data: np.ndarray, C, options: Options):
        halo = 1 # TODO
        grid = data.shape
        GC_data = [
            np.full((grid[0] + 1, grid[1]), C[0]),
            np.full((grid[0], grid[1] + 1), C[1])
        ]
        GC = VectorField(GC_data, halo=halo)
        state = ScalarField(data=data, halo=halo)
        step = make_step(grid, halo, non_unit_g_factor=False, options=options)
        mpdata = MPDATA(step_impl=step, advectee=state, advector=GC)
        return mpdata

    @staticmethod
    def stream_function_2d_basic(grid, size, dt, stream_function, field, options: Options):
        halo = 1 # TODO
        step = make_step(grid, halo, non_unit_g_factor=False, options=options)
        GC = nondivergent_vector_field_2d(grid, size, dt, stream_function, halo)
        advectee = ScalarField(field, halo=halo)
        return MPDATA(step, advectee=advectee, advector=GC)

    @staticmethod
    def stream_function_2d(grid, size, dt, stream_function, field_values, g_factor, options: Options):
        halo = 1 # TODO
        step = make_step(grid, halo, non_unit_g_factor=True, options=options)
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


def from_cdf_1d(cdf: callable, x_min: float, x_max: float, nx: int):
    dx = (x_max - x_min) / nx
    x = np.linspace(x_min + dx / 2, x_max - dx / 2, nx)
    xh = np.linspace(x_min, x_max, nx + 1)
    y = np.diff(cdf(xh)) / dx
    return x, y


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


def make_step(grid, halo, non_unit_g_factor, options):


    n_dims = len(grid)
    ni = grid[0]
    nj = grid[1] if n_dims > 1 else 0
    n_iters = options.n_iters


    if n_dims == 1:
        at = at_1d
        atv0 = atv_1d
        atv1 = atv_1d
        set = set_1d
        get = get_1d
    elif n_dims == 2:
        at = at_2d
        atv0 = atv_2d_0
        atv1 = atv_2d_1
        set = set_2d
        get = get_2d
    else:
        raise NotImplementedError

    @numba.njit(**jit_flags)
    def apply_vector(
            loop, fun0_0, fun0_1, fun1_0, fun1_1,
            out_flag, out_0, out_1,
            arg1_flag, arg1,
            arg2_flag, arg2_0, arg2_1
        ):
        boundary_cond(arg1_flag, arg1, cyclic)
        boundary_cond_vector(arg2_flag, arg2_0, arg2_1, cyclic)

        apply_vector_impl(
            loop, fun0_0, fun0_1, fun1_0, fun1_1,
            out_0, out_1,
            arg1,
            arg2_0, arg2_1
        )
        out_flag[0] = False

    @numba.njit(**jit_flags)
    def apply_vector_impl(loop, fun0_0, fun0_1, fun1_0, fun1_1,
                          out_0, out_1,
                          arg1,
                          arg2_0, arg2_1
                          ):
        out_tpl = (out_0, out_1)
        arg2 = (arg2_0, arg2_1)

        # -1, -1
        if not loop:
            for i in range(halo-1, ni+1+halo-1):
                for j in range(halo-1, nj+1+halo-1) if n_dims > 1 else [-1]:
                    focus = (0, i, j)
                    set(out_tpl[0], i, j, fun0_0((focus, arg1), (focus, arg2)))
                    if n_dims > 1:
                        focus = (1, i, j)
                        set(out_tpl[1], i, j, fun0_1((focus, arg1), (focus, arg2)))
        else:
            for i in range(halo-1, ni+1+halo-1):
                for j in range(halo-1, nj+1+halo-1) if n_dims > 1 else [-1]:
                    focus = (0, i, j)
                    # for axis in range(n_dims):  # TODO: check if loop does not slow down
                    set(out_tpl[0], i, j, fun0_0((focus, arg1), (focus, arg2)))
                    if n_dims > 1:
                        set(out_tpl[0], i, j, fun1_0((focus, arg1), (focus, arg2)))
                    if n_dims > 1:
                        focus = (1, i, j)
                        # for axis in range(n_dims):  # TODO: check if loop does not slow down
                        set(out_tpl[1], i, j, fun0_1((focus, arg1), (focus, arg2)))
                        if n_dims > 1:
                            set(out_tpl[1], i, j, fun1_1((focus, arg1), (focus, arg2)))

    @numba.njit(**jit_flags)
    def apply_scalar(fun_0, fun_1,
                     out_flag, out,
                     arg1_flag, arg1_0, arg1_1,
                     g_factor):
        boundary_cond_vector(arg1_flag, arg1_0, arg1_1, cyclic)
        apply_scalar_impl(fun_0, fun_1, out, arg1_0, arg1_1, g_factor)
        out_flag[0] = False

    @numba.njit(**jit_flags)
    def apply_scalar_impl(fun_0, fun_1,
                     out,
                     arg1_0, arg1_1,
                     g_factor
                     ):
        arg1_tpl = (arg1_0, arg1_1)
        for i in range(halo, ni+halo):
            for j in range(halo, nj+halo) if n_dims > 1 else [-1]:
                focus = (0, i, j)
                set(out, i, j, fun_0(get(out, i, j), (focus, arg1_tpl), (focus, g_factor)))
                if n_dims > 1:
                    focus = (1, i, j)
                    set(out, i, j, fun_1(get(out, i, j), (focus, arg1_tpl), (focus, g_factor)))

    formula_flux_first_pass_0 = make_flux_first_pass(atv0, at)
    formula_flux_subsequent_0 = make_flux_subsequent(atv0, at, infinite_gauge=options.infinite_gauge)
    formula_upwind_0 = make_upwind(atv0, at, non_unit_g_factor)
    formula_antidiff0_0 = make_antidiff(atv0, at, infinite_gauge=options.infinite_gauge, epsilon=options.epsilon, n_dims=n_dims, axis=0)
    formula_antidiff1_0 = make_antidiff(atv0, at, infinite_gauge=options.infinite_gauge, epsilon=options.epsilon, n_dims=n_dims, axis=1)

    formula_flux_first_pass_1 = make_flux_first_pass(atv1, at)
    formula_flux_subsequent_1 = make_flux_subsequent(atv1, at, infinite_gauge=options.infinite_gauge)
    formula_upwind_1 = make_upwind(atv1, at, non_unit_g_factor)
    formula_antidiff0_1 = make_antidiff(atv1, at, infinite_gauge=options.infinite_gauge, epsilon=options.epsilon,
                                      n_dims=n_dims, axis=0)
    formula_antidiff1_1 = make_antidiff(atv1, at, infinite_gauge=options.infinite_gauge, epsilon=options.epsilon,
                                      n_dims=n_dims, axis=1)


    @numba.njit(**jit_flags)
    def boundary_cond_vector(halo_valid, comp_0, comp_1, fun):
        if halo_valid[0]:
            return
        # TODO comp_0[i, :] and comp_1[:, j] not filled
        for j in range(0, halo) if n_dims > 1 else [-1]:
            for i in range(0, ni + 1 + 2 * (halo - 1)):
                focus = (1, i, j)
                set(comp_0, i, j, fun((focus, comp_0), nj, 1))
        for j in range(nj + halo, nj + 2 * halo) if n_dims > 1 else [-1]:
            for i in range(0, ni + 1 + 2 * (halo - 1)):
                focus = (1, i, j)
                set(comp_0, i, j, fun((focus, comp_0), nj, -1))
        if n_dims > 1:
            for i in range(0, halo):
                for j in range(0, nj + 1 + 2 * (halo - 1)):
                    focus = (0, i, j)
                    set(comp_1, i, j, fun((focus, comp_1), ni, 1))
            for i in range(ni+halo, ni+2*halo):
                for j in range(0, nj + 1 + 2 * (halo - 1)):
                    focus = (0, i, j)
                    set(comp_1, i, j, fun((focus, comp_1), ni, -1))




        halo_valid[0] = True

    @numba.njit(**jit_flags)
    def boundary_cond(halo_valid, psi, fun):
        if halo_valid[0]:
            return

        for i in range(0, halo):
            for j in range(0, nj+2*halo) if n_dims > 1 else [-1]:
                focus = (0, i, j)
                set(psi, i, j, fun((focus, psi), ni, 1))
        for i in range(ni+halo, ni+2*halo):
            for j in range(0, nj+2*halo) if n_dims > 1 else [-1]:
                focus = (0, i, j)
                set(psi, i, j, fun((focus, psi), ni, -1))
        if n_dims > 1:
            for j in range(0, halo):
                for i in range(0, ni + 2 * halo):
                    focus = (1, i, j)
                    set(psi, i, j, fun((focus, psi), nj, 1))
            for j in range(nj+halo, nj+2*halo):
                for i in range(0, ni + 2 * halo):
                    focus = (1, i, j)
                    set(psi, i, j, fun((focus, psi), nj, -1))

        halo_valid[0] = True

    @numba.njit(**jit_flags)
    def cyclic(psi, n, sign):
        return at(*psi, sign*n, 0)

    @numba.njit(**jit_flags)
    def step(nt, psi, flux, GC_phys, GC_anti, g_factor):
        for _ in range(nt):
            for it in range(n_iters):
                if it == 0:
                    apply_vector(False, formula_flux_first_pass_0, formula_flux_first_pass_1, formula_flux_first_pass_0, formula_flux_first_pass_1, *flux, *psi, *GC_phys)
                else:
                    if it == 1:
                        apply_vector(True, formula_antidiff0_0, formula_antidiff0_1, formula_antidiff1_0, formula_antidiff1_1, *GC_anti, *psi, *GC_phys)
                        apply_vector(False, formula_flux_subsequent_0, formula_flux_subsequent_1, formula_flux_subsequent_0, formula_flux_subsequent_1, *flux, *psi, *GC_anti)
                    else:
                        apply_vector(True, formula_antidiff0_0, formula_antidiff0_1, formula_antidiff1_0, formula_antidiff1_1, *flux, *psi, *GC_anti)
                        apply_vector(False, formula_flux_subsequent_0, formula_flux_subsequent_1, formula_flux_subsequent_0, formula_flux_subsequent_1, *flux, *psi, *flux)
                    apply_scalar(formula_upwind_0, formula_upwind_1, *psi, *flux, g_factor)
    return step


import numba
import numpy as np
from PyMPDATA_examples.Jarecka_et_al_2015 import formulae

from PyMPDATA import ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Constant
from PyMPDATA.impl.enumerations import ARG_DATA, MAX_DIM_NUM
from PyMPDATA.impl.formulae_divide import make_divide_or_zero


def make_rhs_indexers(ats, grid_step, time_step, options):
    @numba.njit(**options.jit_flags)
    def rhs(m, _0, h, _1, _2, _3):
        return m - (ats(*h, +1) - ats(*h, -1)) / 4 * ats(*h, 0) * time_step / grid_step

    return rhs


def make_rhs(grid_step, time_step, options, traversals):
    indexers = traversals.indexers[traversals.n_dims]
    apply_scalar = traversals.apply_scalar(loop=False)

    formulae_rhs = tuple(
        (
            make_rhs_indexers(
                ats=indexers.ats[i],
                grid_step=grid_step[i],
                time_step=time_step,
                options=options,
            )
            if indexers.ats[i] is not None
            else None
        )
        for i in range(MAX_DIM_NUM)
    )

    @numba.njit(**options.jit_flags)
    def apply(traversals_data, momentum, h):
        null_scalarfield, null_scalarfield_bc = traversals_data.null_scalar_field
        null_vectorfield, null_vectorfield_bc = traversals_data.null_vector_field
        return apply_scalar(
            *formulae_rhs,
            *momentum.field,
            *null_vectorfield,
            null_vectorfield_bc,
            *h.field,
            h.bc,
            *null_scalarfield,
            null_scalarfield_bc,
            *null_scalarfield,
            null_scalarfield_bc,
            *null_scalarfield,
            null_scalarfield_bc,
            traversals_data.buffer
        )

    return apply


def make_interpolate_indexers(ati, options):
    @numba.njit(**options.jit_flags)
    def interpolate(momentum_x, _, momentum_y):
        momenta = (momentum_x[0], (momentum_x[1], momentum_y[1]))
        return ati(*momenta, 0.5)

    return interpolate


def make_interpolate(options, traversals):
    indexers = traversals.indexers[traversals.n_dims]
    apply_vector = traversals.apply_vector()

    formulae_interpolate = tuple(
        (
            make_interpolate_indexers(ati=indexers.ati[i], options=options)
            if indexers.ati[i] is not None
            else None
        )
        for i in range(MAX_DIM_NUM)
    )

    @numba.njit(**options.jit_flags)
    def apply(traversals_data, momentum_x, momentum_y, advector):
        null_scalarfield, null_scalarfield_bc = traversals_data.null_scalar_field
        null_vectorfield, null_vectorfield_bc = traversals_data.null_vector_field
        return apply_vector(
            *formulae_interpolate,
            *advector.field,
            *momentum_x.field,
            momentum_x.bc,
            *null_vectorfield,
            null_vectorfield_bc,
            *momentum_y.field,
            momentum_y.bc,
            traversals_data.buffer
        )

    return apply


class Simulation:
    # pylint: disable=too-few-public-methods
    def __init__(self, settings):
        self.settings = settings
        s = settings

        time_step = s.dt
        halo = settings.options.n_halo
        grid = (s.nx, s.ny)
        bcs = [Constant(value=0)] * len(grid)

        self.advector = VectorField(
            (np.zeros((s.nx + 1, s.ny)), np.zeros((s.nx, s.ny + 1))), halo, bcs
        )

        xi, yi = np.indices(grid, dtype=float)
        xi -= (s.nx - 1) / 2
        yi -= (s.ny - 1) / 2
        x = xi * s.dx
        y = yi * s.dy
        h0 = formulae.amplitude(x, y, s.lx0, s.ly0)

        self.advectees = {
            "h": ScalarField(h0, halo, bcs),
            "uh": ScalarField(np.zeros(grid), halo, bcs),
            "vh": ScalarField(np.zeros(grid), halo, bcs),
        }

        stepper = Stepper(options=s.options, grid=grid)

        grid_step = (s.dx, None, s.dy)
        interpolate = make_interpolate(settings.options, stepper.traversals)
        divide_or_zero = make_divide_or_zero(settings.options, stepper.traversals)
        rhs = make_rhs(grid_step, time_step, settings.options, stepper.traversals)
        traversals_data = stepper.traversals.data

        @numba.experimental.jitclass([])
        class AnteStep:
            def __init__(self):
                pass

            def call(
                self,
                advectees,
                advector,
                step,
                index,
                todo_outer,
                todo_mid3d,
                todo_inner,
            ):
                if index == 0:
                    divide_or_zero(
                        *todo_outer.field,
                        *todo_mid3d.field,
                        *todo_inner.field,
                        *advectees[1].field,
                        *todo_mid3d.field,
                        *advectees[2].field,
                        *advectees[0].field,
                        time_step,
                        grid_step
                    )
                    interpolate(traversals_data, todo_outer, todo_inner, advector)
                else:
                    rhs(traversals_data, advectees[index], advectees[0])

        self.ante_step = AnteStep()

        @numba.experimental.jitclass([])
        class PostStep:
            def __init__(self):
                pass

            def call(self, advectees, step, index):
                if index != 0:
                    rhs(traversals_data, advectees[index], advectees[0])

        self.post_step = PostStep()
        self.solver = Solver(stepper, self.advectees, self.advector)

    def run(self):
        s = self.settings
        output = []
        for it in range(100):
            if it != 0:
                self.solver.advance(
                    1, ante_step=self.ante_step, post_step=self.post_step
                )
            output.append(
                {k: self.solver.advectee[k].get().copy() for k in self.advectees.keys()}
            )
        return output

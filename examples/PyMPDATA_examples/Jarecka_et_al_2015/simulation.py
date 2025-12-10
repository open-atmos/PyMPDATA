import numba
import numpy as np
from PyMPDATA_examples.Jarecka_et_al_2015 import formulae

from PyMPDATA import ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Constant
from PyMPDATA.impl.enumerations import INNER, OUTER
from PyMPDATA.impl.formulae_divide import make_divide_or_zero


def make_hooks(*, traversals, options, grid_step, time_step):

    divide_or_zero = make_divide_or_zero(options, traversals)
    interpolate = formulae.make_interpolate(options, traversals)
    rhs_x = formulae.make_rhs(grid_step, time_step, OUTER, options, traversals)
    rhs_y = formulae.make_rhs(grid_step, time_step, INNER, options, traversals)

    @numba.experimental.jitclass([])
    class AnteStep:  # pylint:disable=too-few-public-methods
        def __init__(self):
            pass

        def call(
            self,
            traversals_data,
            advectees,
            advector,
            _,
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
            elif index == 1:
                rhs_x(traversals_data, advectees[index], advectees[0])
            else:
                rhs_y(traversals_data, advectees[index], advectees[0])

    @numba.experimental.jitclass([])
    class PostStep:  # pylint:disable=too-few-public-methods
        def __init__(self):
            pass

        def call(self, traversals_data, advectees, _, index):
            if index == 0:
                pass
            if index == 1:
                rhs_x(traversals_data, advectees[index], advectees[0])
            else:
                rhs_y(traversals_data, advectees[index], advectees[0])

    return AnteStep(), PostStep()


class Simulation:
    # pylint: disable=too-few-public-methods
    def __init__(self, settings):
        self.settings = settings
        s = settings

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

        self.ante_step, self.post_step = make_hooks(
            traversals=stepper.traversals,
            options=settings.options,
            grid_step=(s.dx, None, s.dy),
            time_step=s.dt,
        )

        self.solver = Solver(stepper, self.advectees, self.advector)

    def run(self):
        s = self.settings
        output = []
        for it in range(s.nt + 1):
            if it != 0:
                self.solver.advance(
                    1, ante_step=self.ante_step, post_step=self.post_step
                )
            if it % s.outfreq == 0:
                output.append(
                    {
                        k: self.solver.advectee[k].get().copy()
                        for k in self.advectees.keys()  # pylint:disable=consider-iterating-dictionary
                    }
                )
        return output

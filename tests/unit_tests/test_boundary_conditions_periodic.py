import numpy as np
from PyMPDATA import ScalarField, VectorField, Stepper, Solver, Options
from PyMPDATA.boundary_conditions import Periodic

options = Options(divergent_flow=True)
assert options.n_halo == 2


class Test1D:
    size = 5
    bcs = (Periodic(),)

    def test_single_step_scalar(self):
        # arrange
        scl = ScalarField(np.arange(self.size, dtype=float), options.n_halo, self.bcs)
        vct = VectorField((np.zeros(self.size+1, dtype=float),), options.n_halo, self.bcs)

        stepper = Stepper(options=options, n_dims=1)
        solver = Solver(stepper, scl, vct)

        # act
        solver.advance(1)

        # assert - left halo
        np.testing.assert_array_equal(
            scl.data[0:options.n_halo],
            scl.data[self.size: options.n_halo + self.size]
        )
        # assert - right halo
        np.testing.assert_array_equal(
            scl.data[-options.n_halo:],
            scl.data[options.n_halo: 2 * options.n_halo]
        )

    def test_single_step_vector(self):
        # arrange
        scl = ScalarField(np.zeros(self.size, dtype=float), options.n_halo, self.bcs)
        vct = VectorField((np.arange(self.size+1, dtype=float) + .5,), options.n_halo, self.bcs)

        stepper = Stepper(options=options, n_dims=1)
        solver = Solver(stepper, scl, vct)

        # act
        solver.advance(1)

        # assert - left halo
        np.testing.assert_array_equal(
            vct.data[0][0:options.n_halo-1],
            vct.data[0][self.size: options.n_halo + self.size - 1]
        )
        # assert - right halo
        np.testing.assert_array_equal(
            vct.data[0][-options.n_halo + 1:],
            vct.data[0][options.n_halo: 2 * (options.n_halo-1) + 1]
        )


class Test2D:
    size = (2, 3)
    bcs = (Periodic(), Periodic())

    def test_single_step_scalar(self):
        # arrange
        scl = ScalarField(
            np.arange(np.prod(self.size), dtype=float).reshape(self.size),
            options.n_halo,
            self.bcs
        )
        vct = VectorField(
            (
                np.zeros(
                    (self.size[0] + 1, self.size[1]),
                    dtype=float
                ),
                np.zeros(
                    (self.size[0], self.size[1] + 1),
                    dtype=float
                ),
            ),
            options.n_halo,
            self.bcs
        )

        stepper = Stepper(options=options, n_dims=len(self.size))
        solver = Solver(stepper, scl, vct)

        # act
        solver.advance(1)

        # assert - left halo
        np.testing.assert_array_equal(
            scl.data[0:options.n_halo, :],
            scl.data[self.size[0]: options.n_halo + self.size[0], :]
        )
        # assert - right halo
        np.testing.assert_array_equal(
            scl.data[-options.n_halo:, :],
            scl.data[options.n_halo: 2 * options.n_halo, :]
        )
        # assert - lower halo
        # TODO

        # assert - upper halo
        # TODO

    def test_single_step_vector(self):
        # TODO
        pass
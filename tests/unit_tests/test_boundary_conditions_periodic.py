# pylint: disable=missing-function-docstring,line-too-long,missing-class-docstring,missing-module-docstring
import numpy as np
import pytest

from PyMPDATA import Options, ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Periodic
from PyMPDATA.impl.enumerations import INNER, OUTER

options = Options(divergent_flow=True)
assert options.n_halo == 2


class Test1D:
    size = 5
    bcs = (Periodic(),)

    def test_single_step_scalar(self):
        # arrange
        scl = ScalarField(np.arange(self.size, dtype=float), options.n_halo, self.bcs)
        vct = VectorField(
            (np.zeros(self.size + 1, dtype=float),), options.n_halo, self.bcs
        )

        stepper = Stepper(options=options, n_dims=1)
        solver = Solver(stepper, scl, vct)

        # act
        solver.advance(1)

        # assert - left halo
        np.testing.assert_array_equal(
            scl.data[0 : options.n_halo],
            scl.data[self.size : options.n_halo + self.size],
        )
        # assert - right halo
        np.testing.assert_array_equal(
            scl.data[-options.n_halo :], scl.data[options.n_halo : 2 * options.n_halo]
        )

    def test_single_step_vector(self):
        # arrange
        scl = ScalarField(np.zeros(self.size, dtype=float), options.n_halo, self.bcs)
        vct = VectorField(
            (np.arange(self.size + 1, dtype=float) + 0.5,), options.n_halo, self.bcs
        )

        stepper = Stepper(options=options, n_dims=1)
        solver = Solver(stepper, scl, vct)

        # act
        solver.advance(1)

        # assert - left halo
        np.testing.assert_array_equal(
            vct.data[OUTER][0 : options.n_halo - 1],
            vct.data[OUTER][self.size : options.n_halo + self.size - 1],
        )
        # assert - right halo
        np.testing.assert_array_equal(
            vct.data[OUTER][-options.n_halo + 1 :],
            vct.data[OUTER][options.n_halo : 2 * (options.n_halo - 1) + 1],
        )


@pytest.mark.parametrize("threading", (False, True))
class Test2D:
    size = (3, 2)
    bcs = (Periodic(), Periodic())

    def test_single_step_scalar(self, threading):
        # arrange
        scl = ScalarField(
            np.arange(np.prod(self.size), dtype=float).reshape(self.size),
            options.n_halo,
            self.bcs,
        )
        vct = VectorField(
            (
                np.zeros((self.size[OUTER] + 1, self.size[INNER]), dtype=float),
                np.zeros((self.size[OUTER], self.size[INNER] + 1), dtype=float),
            ),
            options.n_halo,
            self.bcs,
        )

        stepper_ctor_args = {"options": options, "n_dims": len(self.size)}
        if not threading:
            stepper_ctor_args["n_threads"] = 1
        stepper = Stepper(**stepper_ctor_args)
        solver = Solver(stepper, scl, vct)

        # act
        solver.advance(1)

        # assert - left halo
        np.testing.assert_array_equal(
            scl.data[0 : options.n_halo, :],
            scl.data[self.size[OUTER] : options.n_halo + self.size[OUTER], :],
        )
        # assert - right halo
        np.testing.assert_array_equal(
            scl.data[-options.n_halo :, :],
            scl.data[options.n_halo : 2 * options.n_halo, :],
        )
        # assert - lower halo
        np.testing.assert_array_equal(
            scl.data[:, 0 : options.n_halo],
            scl.data[:, self.size[INNER] : options.n_halo + self.size[INNER]],
        )
        # assert - upper halo
        np.testing.assert_array_equal(
            scl.data[:, -options.n_halo :],
            scl.data[:, options.n_halo : 2 * options.n_halo],
        )

    def test_single_step_vector(self, threading):
        # arrange
        scl = ScalarField(
            np.zeros(np.prod(self.size), dtype=float).reshape(self.size),
            options.n_halo,
            self.bcs,
        )
        grid_0 = (self.size[OUTER] + 1, self.size[INNER])
        grid_1 = (self.size[OUTER], self.size[INNER] + 1)
        vct = VectorField(
            (
                np.arange(np.product(grid_0), dtype=float).reshape(grid_0),
                np.arange(np.product(grid_1), dtype=float).reshape(grid_1),
            ),
            options.n_halo,
            self.bcs,
        )

        stepper_ctor_args = {"options": options, "n_dims": len(self.size)}
        if not threading:
            stepper_ctor_args["n_threads"] = 1
        stepper = Stepper(**stepper_ctor_args)
        solver = Solver(stepper, scl, vct)

        # act
        solver.advance(1)

        outer_inner_domain_excl_halo = slice(
            options.n_halo, vct.data[OUTER].shape[INNER] - options.n_halo
        )
        outer_outer_domain_excl_halo = slice(
            options.n_halo - 1, vct.data[OUTER].shape[OUTER] - (options.n_halo - 1)
        )
        inner_inner_domain_excl_halo = slice(
            options.n_halo - 1, vct.data[INNER].shape[INNER] - (options.n_halo - 1)
        )
        inner_outer_domain_excl_halo = slice(
            options.n_halo, options.n_halo + vct.data[INNER].shape[OUTER]
        )
        # ====== OUTER ======
        # assert - left halo
        np.testing.assert_array_equal(
            vct.data[OUTER][0 : options.n_halo - 1, outer_inner_domain_excl_halo],
            vct.data[OUTER][
                self.size[OUTER] : options.n_halo + self.size[OUTER] - 1,
                outer_inner_domain_excl_halo,
            ],
        )
        # assert - right halo
        np.testing.assert_array_equal(
            vct.data[OUTER][-options.n_halo + 1 :, outer_inner_domain_excl_halo],
            vct.data[OUTER][
                options.n_halo : 2 * (options.n_halo - 1) + 1,
                outer_inner_domain_excl_halo,
            ],
        )
        # assert - lower halo
        np.testing.assert_array_equal(
            vct.data[OUTER][outer_outer_domain_excl_halo, 0 : options.n_halo],
            vct.data[OUTER][
                outer_outer_domain_excl_halo,
                self.size[INNER] : options.n_halo + self.size[INNER],
            ],
        )
        # assert - upper halo
        np.testing.assert_array_equal(
            vct.data[OUTER][outer_outer_domain_excl_halo, -options.n_halo :],
            vct.data[OUTER][
                outer_outer_domain_excl_halo, options.n_halo : 2 * options.n_halo
            ],
        )

        # ====== INNER ======
        # assert - left halo
        np.testing.assert_array_equal(
            vct.data[INNER][0 : options.n_halo, inner_inner_domain_excl_halo],
            vct.data[INNER][
                self.size[OUTER] : options.n_halo + self.size[OUTER],
                inner_inner_domain_excl_halo,
            ],
        )
        # assert - right halo
        np.testing.assert_array_equal(
            vct.data[INNER][-options.n_halo :, inner_inner_domain_excl_halo],
            vct.data[INNER][
                options.n_halo : 2 * options.n_halo, inner_inner_domain_excl_halo
            ],
        )
        # # assert - lower halo
        np.testing.assert_array_equal(
            vct.data[INNER][inner_outer_domain_excl_halo, 0 : options.n_halo - 1],
            vct.data[INNER][
                inner_outer_domain_excl_halo,
                self.size[INNER] : options.n_halo + self.size[INNER] - 1,
            ],
        )
        # # assert - upper halo
        np.testing.assert_array_equal(
            vct.data[INNER][
                inner_outer_domain_excl_halo, self.size[INNER] + options.n_halo :
            ],
            vct.data[INNER][
                inner_outer_domain_excl_halo,
                self.size[INNER] : options.n_halo + self.size[INNER] - 1,
            ],
        )

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from scipy import interpolate

from PyMPDATA import Options, ScalarField, VectorField
from PyMPDATA.boundary_conditions import Extrapolated
from PyMPDATA.impl.traversals import Traversals

JIT_FLAGS = Options().jit_flags


class TestBoundaryConditionExtrapolated:
    @staticmethod
    @pytest.mark.parametrize("halo", (1, 2, 3, 4))
    @pytest.mark.parametrize(
        "data",
        (
            np.array([11, 12, 13, 14], dtype=float),
            np.array([11, 12, 13, 14], dtype=complex),
            np.array([1, 2, 3, 4], dtype=float),
            np.array([1, 2, 3, 4], dtype=complex),
        ),
    )
    def test_1d_scalar(data, halo, n_threads=1, left_first=True):
        # arrange
        boundary_conditions = (Extrapolated(),)
        field = ScalarField(data, halo, boundary_conditions)
        # pylint:disable=duplicate-code
        traversals = Traversals(
            grid=field.grid,
            halo=halo,
            jit_flags=JIT_FLAGS,
            n_threads=n_threads,
            left_first=left_first,
        )
        field.assemble(traversals)
        meta_and_data, fill_halos = field.impl
        sut = traversals._code["fill_halos_scalar"]  # pylint:disable=protected-access

        # act
        thread_id = 0
        sut(thread_id, *meta_and_data, fill_halos)

        # assert
        extrapolator = interpolate.interp1d(
            np.linspace(halo, len(data) - 1 + halo, len(data)),
            data,
            fill_value="extrapolate",
        )
        np.testing.assert_array_equal(
            field.data[0:halo], np.maximum(extrapolator(np.arange(halo)), 0)
        )
        np.testing.assert_array_equal(
            field.data[-halo:],
            np.maximum(
                extrapolator(
                    np.linspace(len(data) + halo, len(data) + 2 * halo - 1, halo)
                ),
                0,
            ),
        )

    @staticmethod
    @pytest.mark.parametrize("data", (np.array([0, 2, 3, 0], dtype=float),))
    @pytest.mark.parametrize("halo", (2, 3, 4))
    def test_1d_vector(data, halo, n_threads=1, left_first=True):
        # arrange
        boundary_condition = (Extrapolated(),)
        field = VectorField((data,), halo, boundary_condition)
        # pylint:disable=duplicate-code
        traversals = Traversals(
            grid=field.grid,
            halo=halo,
            jit_flags=JIT_FLAGS,
            n_threads=n_threads,
            left_first=left_first,
        )
        field.assemble(traversals)
        meta_and_data, fill_halos = field.impl
        meta_and_data = (
            meta_and_data[0],
            (meta_and_data[1], meta_and_data[2], meta_and_data[3]),
        )
        sut = traversals._code["fill_halos_vector"]  # pylint:disable=protected-access

        # act
        thread_id = 0
        sut(thread_id, *meta_and_data, fill_halos)

        # assert
        assert (field.data[0][0 : halo - 1] == data[0]).all()
        assert (field.data[0][-(halo - 1) :] == data[-1]).all()

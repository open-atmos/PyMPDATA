# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import warnings
from functools import lru_cache

import numpy as np
import pytest
from numba.core.errors import NumbaExperimentalFeatureWarning

from PyMPDATA import Options, ScalarField, VectorField
from PyMPDATA.boundary_conditions import Periodic
from PyMPDATA.impl.meta import INNER, MID3D, OUTER
from PyMPDATA.impl.traversals import Traversals
from tests.unit_tests.fixtures.n_threads import n_threads

assert hasattr(n_threads, "_pytestfixturefunction")


LEFT, RIGHT = "left", "right"
ALL = (None, None)

DIMENSIONS = (
    pytest.param(OUTER, id="OUTER"),
    pytest.param(MID3D, id="MID3D"),
    pytest.param(INNER, id="INNER"),
)


def shift(tup, num):
    if not tup or not num:
        return tup
    num %= len(tup)
    return tup[num:] + tup[:num]


def indices(arg1, arg2=None, arg3=None):
    if arg2 is not None:
        if arg3 is not None:
            return (
                slice(arg1[0] if arg1[0] else None, arg1[1] if arg1[1] else None),
                slice(arg2[0] if arg2[0] else None, arg2[1] if arg2[1] else None),
                slice(arg3[0] if arg3[0] else None, arg3[1] if arg3[1] else None),
            )
        return (
            slice(arg1[0] if arg1[0] else None, arg1[1] if arg1[1] else None),
            slice(arg2[0] if arg2[0] else None, arg2[1] if arg2[1] else None),
        )
    return slice(arg1[0], arg1[1])


JIT_FLAGS = Options().jit_flags


@lru_cache()
# pylint: disable-next=redefined-outer-name
def make_traversals(grid, halo, n_threads, left_first):
    return Traversals(
        grid=grid,
        halo=halo,
        jit_flags=JIT_FLAGS,
        n_threads=n_threads,
        left_first=left_first,
        buffer_size=0,
    )


class TestPeriodicBoundaryCondition:
    @staticmethod
    @pytest.mark.parametrize(
        "data",
        (
            np.array([1, 2, 3]),
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            np.arange(3 * 4 * 5).reshape((3, 4, 5)),
        ),
    )
    @pytest.mark.parametrize("halo", (1, 2, 3))
    @pytest.mark.parametrize("side", (LEFT, RIGHT))
    @pytest.mark.parametrize("dim", DIMENSIONS)
    # pylint: disable-next=redefined-outer-name,too-many-arguments
    def test_scalar(data, halo, side, n_threads, dim, left_first=True):
        n_dims = len(data.shape)
        if n_dims == 1 and dim != INNER:
            return
        if n_dims == 2 and dim == MID3D:
            return
        if n_dims == 1 and n_threads > 1:
            return

        # arrange
        field = ScalarField(data, halo, tuple(Periodic() for _ in range(n_dims)))
        traversals = make_traversals(
            grid=field.grid, halo=halo, n_threads=n_threads, left_first=left_first
        )
        field.assemble(traversals)
        meta_and_data, fill_halos = field.impl
        sut = traversals._code["fill_halos_scalar"]  # pylint:disable=protected-access

        # act
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaExperimentalFeatureWarning)
            for thread_id in range(
                n_threads
            ):  # TODO #96: xfail if not all threads executed?
                sut(thread_id, *meta_and_data, fill_halos, traversals.data.buffer)

        # assert
        interior = (halo, -halo)
        if side == LEFT:
            np.testing.assert_array_equal(
                field.data[
                    shift(indices((None, halo), interior, interior)[:n_dims], dim)
                ],
                data[shift(indices((-halo, None), ALL, ALL)[:n_dims], dim)],
            )
        else:
            np.testing.assert_array_equal(
                field.data[
                    shift(indices((-halo, None), interior, interior)[:n_dims], dim)
                ],
                data[shift(indices((None, halo), ALL, ALL)[:n_dims], dim)],
            )

    @staticmethod
    @pytest.mark.parametrize(
        "data",
        (
            (np.array([1, 2, 3]),),
            (
                np.array([[41, 42, 33], [51, 52, 53], [61, 62, 63], [71, 72, 73]]),
                np.array([[11, 12, 13, 14], [21, 22, 23, 24], [31, 32, 33, 34]]),
            ),
            (
                np.arange(4 * 4 * 5).reshape((3 + 1, 4, 5)),
                np.arange(3 * 5 * 5).reshape((3, 4 + 1, 5)),
                np.arange(3 * 4 * 6).reshape((3, 4, 5 + 1)),
            ),
        ),
    )
    @pytest.mark.parametrize("halo", (1, 2, 3))
    @pytest.mark.parametrize("side", (LEFT, RIGHT))
    @pytest.mark.parametrize("comp", DIMENSIONS)
    @pytest.mark.parametrize("dim_offset", (0, 1, 2))
    # pylint: disable=redefined-outer-name,too-many-arguments,too-many-branches
    def test_vector(data, halo, side, n_threads, comp, dim_offset, left_first=True):
        n_dims = len(data)
        if n_dims == 1 and n_threads > 1:
            return
        if n_dims == 1 and (comp != INNER or dim_offset != 0):
            return
        if n_dims == 2 and (comp == MID3D or dim_offset == 2):
            return

        # arrange
        field = VectorField(data, halo, tuple(Periodic() for _ in range(n_dims)))
        traversals = make_traversals(
            grid=field.grid, halo=halo, n_threads=n_threads, left_first=left_first
        )
        field.assemble(traversals)
        meta_and_data, fill_halos = field.impl
        meta_and_data = (
            meta_and_data[0],
            (meta_and_data[1], meta_and_data[2], meta_and_data[3]),
        )
        sut = traversals._code["fill_halos_vector"]  # pylint:disable=protected-access

        # act
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaExperimentalFeatureWarning)
            for thread_id in range(
                n_threads
            ):  # TODO #96: xfail if not all threads executed?
                sut(thread_id, *meta_and_data, fill_halos, traversals.data.buffer)

        # assert
        interior = (halo, -halo)
        if n_dims == 1 and halo == 1:
            np.testing.assert_array_equal(field.data[comp], data[comp])
        if side == LEFT:
            if dim_offset == 1:
                np.testing.assert_array_equal(
                    field.data[comp][
                        shift(
                            indices((None, halo), (halo - 1, -(halo - 1)), interior)[
                                :n_dims
                            ],
                            -comp + dim_offset,
                        )
                    ],
                    data[comp][
                        shift(
                            indices((-halo, None), ALL, ALL)[:n_dims],
                            -comp + dim_offset,
                        )
                    ],
                )
            elif dim_offset == 2:
                np.testing.assert_array_equal(
                    field.data[comp][
                        shift(
                            indices((None, halo), interior, (halo - 1, -(halo - 1)))[
                                :n_dims
                            ],
                            -comp + dim_offset,
                        )
                    ],
                    data[comp][
                        shift(
                            indices((-halo, None), ALL, ALL)[:n_dims],
                            -comp + dim_offset,
                        )
                    ],
                )
            elif dim_offset == 0:
                if halo == 1:
                    return
                np.testing.assert_array_equal(
                    field.data[comp][
                        shift(
                            indices((None, halo - 1), interior, interior)[:n_dims],
                            -comp + dim_offset,
                        )
                    ],
                    data[comp][
                        shift(
                            indices((-(halo - 1) - 1, -1), ALL, ALL)[:n_dims],
                            -comp + dim_offset,
                        )
                    ],
                )
            else:
                raise NotImplementedError()
        else:
            if dim_offset == 1:
                np.testing.assert_array_equal(
                    field.data[comp][
                        shift(
                            indices((-halo, None), (halo - 1, -(halo - 1)), interior)[
                                :n_dims
                            ],
                            -comp + dim_offset,
                        )
                    ],
                    data[comp][
                        shift(
                            indices((None, halo), ALL, ALL)[:n_dims], -comp + dim_offset
                        )
                    ],
                )
            elif dim_offset == 2:
                np.testing.assert_array_equal(
                    field.data[comp][
                        shift(
                            indices((-halo, None), interior, (halo - 1, -(halo - 1)))[
                                :n_dims
                            ],
                            -comp + dim_offset,
                        )
                    ],
                    data[comp][
                        shift(
                            indices((None, halo), ALL, ALL)[:n_dims], -comp + dim_offset
                        )
                    ],
                )
            elif dim_offset == 0:
                if halo == 1:
                    return
                np.testing.assert_array_equal(
                    field.data[comp][
                        shift(
                            indices((-(halo - 1), None), interior, interior)[:n_dims],
                            -comp + dim_offset,
                        )
                    ],
                    data[comp][
                        shift(
                            indices((1, halo - 1 + 1), ALL, ALL)[:n_dims],
                            -comp + dim_offset,
                        )
                    ],
                )
            else:
                raise NotImplementedError()

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from functools import lru_cache
import numpy as np
import pytest
from PyMPDATA import ScalarField, VectorField, Options
from PyMPDATA.boundary_conditions import Periodic
from PyMPDATA.impl.traversals import Traversals
from PyMPDATA.impl.meta import OUTER, MID3D, INNER

from tests.unit_tests.fixtures.n_threads import n_threads
assert hasattr(n_threads, '_pytestfixturefunction')


LEFT, RIGHT = 'left', 'right'
ALL = (None, None)

DIMENSIONS = (
    pytest.param(OUTER, id="OUTER"),
    pytest.param(MID3D, id="MID3D"),
    pytest.param(INNER, id="INNER")
)


def shift(tup, n):
    if not tup or not n:
        return tup
    n %= len(tup)
    return tup[n:] + tup[:n]


def indices(a, b=None, c=None):
    if b is not None:
        if c is not None:
            return (
                slice(a[0] if a[0] else None, a[1] if a[1] else None),
                slice(b[0] if b[0] else None, b[1] if b[1] else None),
                slice(c[0] if c[0] else None, c[1] if c[1] else None)
            )
        return (
            slice(a[0] if a[0] else None, a[1] if a[1] else None),
            slice(b[0] if b[0] else None, b[1] if b[1] else None)
        )
    return slice(a[0], a[1])


JIT_FLAGS = Options().jit_flags


@lru_cache()
# pylint: disable-next=redefined-outer-name
def make_traversals(grid, halo, n_threads):
    return Traversals(grid=grid, halo=halo, jit_flags=JIT_FLAGS, n_threads=n_threads)


class TestPeriodicBoundaryCondition:
    @staticmethod
    @pytest.mark.parametrize("data", (
            np.array([1, 2, 3]),
            np.array([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]),
            np.arange(3 * 4 * 5).reshape((3, 4, 5))
    ))
    @pytest.mark.parametrize("halo", (1, 2, 3))
    @pytest.mark.parametrize("side", (LEFT, RIGHT))
    @pytest.mark.parametrize("dim", DIMENSIONS)
    # pylint: disable-next=redefined-outer-name
    def test_scalar(data, halo, side, n_threads, dim):
        n_dims = len(data.shape)
        if n_dims == 1 and dim != INNER:
            return
        if n_dims == 2 and dim == MID3D:
            return
        if n_dims == 1 and n_threads > 1:
            return

        # arrange
        field = ScalarField(data, halo, tuple([Periodic() for _ in range(n_dims)]))
        traversals = make_traversals(grid=field.grid, halo=halo, n_threads=n_threads)
        field.assemble(traversals)
        meta_and_data, fill_halos = field.impl
        sut = traversals._fill_halos_scalar

        # act
        for thread_id in range(n_threads):  # TODO #96: xfail if not all threads executed?
            sut(thread_id, *meta_and_data, *fill_halos)

        # assert
        INT = (halo, -halo)
        if side == LEFT:
            np.testing.assert_array_equal(
                field.data[shift(indices((None, halo), INT, INT)[:n_dims], dim)],
                data[shift(indices((-halo, None), ALL, ALL)[:n_dims], dim)]
            )
        else:
            np.testing.assert_array_equal(
                field.data[shift(indices((-halo, None), INT, INT)[:n_dims], dim)],
                data[shift(indices((None, halo), ALL, ALL)[:n_dims], dim)]
            )

    @staticmethod
    @pytest.mark.parametrize("data", (
            (np.array([1, 2, 3]),),
            (
                np.array([
                    [41, 42, 33],
                    [51, 52, 53],
                    [61, 62, 63],
                    [71, 72, 73]
                ]),
                np.array([
                    [11, 12, 13, 14],
                    [21, 22, 23, 24],
                    [31, 32, 33, 34]
                ]),
            ),
            (
                np.arange(4 * 4 * 5).reshape((3+1, 4, 5)),
                np.arange(3 * 5 * 5).reshape((3, 4+1, 5)),
                np.arange(3 * 4 * 6).reshape((3, 4, 5+1))
            )
    ))
    @pytest.mark.parametrize("halo", (1, 2, 3))
    @pytest.mark.parametrize("side", (LEFT, RIGHT))
    @pytest.mark.parametrize("component", DIMENSIONS)
    @pytest.mark.parametrize("dim_offset", (0, 1, 2))
    # pylint: disable-next=redefined-outer-name
    def test_vector(data, halo, side, n_threads, component, dim_offset):
        n_dims = len(data)
        if n_dims == 1 and n_threads > 1:
            return
        if n_dims == 1 and (component != INNER or dim_offset != 0):
            return
        if n_dims == 2 and (component == MID3D or dim_offset == 2):
            return

        # arrange
        field = VectorField(data, halo, tuple([Periodic() for _ in range(n_dims)]))
        traversals = make_traversals(grid=field.grid, halo=halo, n_threads=n_threads)
        field.assemble(traversals)
        meta_and_data, fill_halos = field.impl
        sut = traversals._fill_halos_vector

        # act
        for thread_id in range(n_threads):  # TODO #96: xfail if not all threads executed?
            sut(thread_id, *meta_and_data, *fill_halos)

        # assert
        INT = (halo, -halo)
        if n_dims == 1 and halo == 1:
            np.testing.assert_array_equal(field.data[component], data[component])
        if side == LEFT:
            if dim_offset == 1:
                np.testing.assert_array_equal(
                    field.data[component][
                        shift(
                            indices((None, halo), (halo - 1, -(halo - 1)), INT)[:n_dims],
                            -component+dim_offset
                        )
                    ],
                    data[component][
                        shift(
                            indices((-halo, None), ALL, ALL)[:n_dims],
                            -component+dim_offset
                        )
                    ]
                )
            elif dim_offset == 2:
                np.testing.assert_array_equal(
                    field.data[component][
                        shift(
                            indices((None, halo), INT, (halo - 1, -(halo - 1)))[:n_dims],
                            -component+dim_offset
                        )
                    ],
                    data[component][shift(indices((-halo, None), ALL, ALL)[:n_dims], -component+dim_offset)]
                )
            elif dim_offset == 0:
                np.testing.assert_array_equal(
                    field.data[component][shift(
                        indices((None, halo - 1), INT, INT)[:n_dims],
                        -component+dim_offset
                    )],
                    data[component][shift(
                        indices((-(halo - 1), None), ALL, ALL)[:n_dims],
                        -component+dim_offset
                    )]
                )
        else:
            if dim_offset == 1:
                np.testing.assert_array_equal(
                    field.data[component][
                        shift(
                            indices((-halo, None), (halo - 1, -(halo - 1)), INT)[:n_dims],
                            -component+dim_offset
                        )
                    ],
                    data[component][
                        shift(
                            indices((None, halo), ALL, ALL)[:n_dims],
                            -component+dim_offset
                        )
                    ]
                )
            elif dim_offset == 2:
                np.testing.assert_array_equal(
                    field.data[component][
                        shift(
                            indices((-halo, None), INT, (halo - 1, -(halo - 1)))[:n_dims],
                            -component+dim_offset
                        )
                    ],
                    data[component][
                        shift(
                            indices((None, halo), ALL, ALL)[:n_dims],
                            -component+dim_offset
                        )
                    ]
                )
            elif dim_offset == 0:
                np.testing.assert_array_equal(
                    field.data[component][
                        shift(
                            indices((-(halo - 1), None), INT, INT)[:n_dims],
                            -component+dim_offset
                        )
                    ],
                    data[component][
                        shift(
                            indices((None, halo - 1), ALL, ALL)[:n_dims],
                            -component+dim_offset
                        )
                    ]
                )

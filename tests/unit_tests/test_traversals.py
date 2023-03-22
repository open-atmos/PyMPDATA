# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import warnings
from collections import namedtuple
from functools import lru_cache

import numba
import numpy as np
import pytest
from numba.core.errors import NumbaExperimentalFeatureWarning

from PyMPDATA import Options, ScalarField, VectorField
from PyMPDATA.boundary_conditions import Constant
from PyMPDATA.impl.enumerations import (
    ARG_FOCUS,
    IMPL_BC,
    IMPL_META_AND_DATA,
    INNER,
    INVALID_INDEX,
    MAX_DIM_NUM,
    META_AND_DATA_META,
    MID3D,
    OUTER,
)
from PyMPDATA.impl.meta import META_HALO_VALID
from PyMPDATA.impl.traversals import Traversals
from tests.unit_tests.fixtures.n_threads import n_threads

assert hasattr(n_threads, "_pytestfixturefunction")

jit_flags = Options().jit_flags


@lru_cache()
# pylint: disable-next=redefined-outer-name
def make_traversals(grid, halo, n_threads, left_first):
    return Traversals(
        grid=grid,
        halo=halo,
        jit_flags=jit_flags,
        n_threads=n_threads,
        left_first=left_first,
    )


@numba.njit(**jit_flags)
def cell_id(i, j, k):
    if i == INVALID_INDEX:
        i = 0
    if j == INVALID_INDEX:
        j = 0
    return 100 * i + 10 * j + k


@numba.njit(**jit_flags)
# pylint: disable=too-many-arguments
def _cell_id_scalar(value, arg_1_vec, arg_2_scl, arg_3_scl, arg_4_scl, arg_5_scl):
    focus = arg_1_vec[ARG_FOCUS]
    if focus != arg_2_scl[ARG_FOCUS]:
        raise ValueError()
    if focus != arg_3_scl[ARG_FOCUS]:
        raise ValueError()
    if focus != arg_4_scl[ARG_FOCUS]:
        raise ValueError()
    if focus != arg_5_scl[ARG_FOCUS]:
        raise ValueError()
    return value + cell_id(*focus)


@numba.njit(**jit_flags)
def _cell_id_vector(arg_1, arg_2, arg_3):
    focus = arg_1[ARG_FOCUS]
    if focus != arg_2[ARG_FOCUS]:
        raise ValueError()
    if focus != arg_3[ARG_FOCUS]:
        raise ValueError()
    return cell_id(*focus)


def make_commons(grid, halo, num_threads, left_first):
    traversals = make_traversals(grid, halo, num_threads, left_first)
    n_dims = len(grid)
    halos = ((halo - 1, halo, halo), (halo, halo - 1, halo), (halo, halo, halo - 1))

    _Commons = namedtuple(
        "_Commons",
        ("n_dims", "traversals", "scl_null_arg_impl", "vec_null_arg_impl", "halos"),
    )

    return _Commons(
        n_dims=n_dims,
        traversals=traversals,
        scl_null_arg_impl=ScalarField.make_null(n_dims, traversals).impl,
        vec_null_arg_impl=VectorField.make_null(n_dims, traversals).impl,
        halos=halos,
    )


class TestTraversals:
    @staticmethod
    @pytest.mark.parametrize("halo", (1, 2, 3))
    @pytest.mark.parametrize("grid", ((3, 4, 5), (5, 6), (11,)))
    @pytest.mark.parametrize("loop", (True, False))
    # pylint:disable=redefined-outer-name
    def test_apply_scalar(
        n_threads,
        halo: int,
        grid: tuple,
        loop: bool,
        left_first: bool = True,
    ):
        if len(grid) == 1 and n_threads > 1:
            return
        cmn = make_commons(grid, halo, n_threads, left_first)

        # arrange
        sut = cmn.traversals.apply_scalar(loop=loop)
        out = ScalarField(np.zeros(grid), halo, tuple([Constant(np.nan)] * cmn.n_dims))
        out.assemble(cmn.traversals)

        # act
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaExperimentalFeatureWarning)
            sut(
                _cell_id_scalar,
                _cell_id_scalar if loop else None,
                _cell_id_scalar if loop else None,
                *out.impl[IMPL_META_AND_DATA],
                *cmn.vec_null_arg_impl[IMPL_META_AND_DATA],
                *cmn.vec_null_arg_impl[IMPL_BC],
                *cmn.scl_null_arg_impl[IMPL_META_AND_DATA],
                *cmn.scl_null_arg_impl[IMPL_BC],
                *cmn.scl_null_arg_impl[IMPL_META_AND_DATA],
                *cmn.scl_null_arg_impl[IMPL_BC],
                *cmn.scl_null_arg_impl[IMPL_META_AND_DATA],
                *cmn.scl_null_arg_impl[IMPL_BC],
                *cmn.scl_null_arg_impl[IMPL_META_AND_DATA],
                *cmn.scl_null_arg_impl[IMPL_BC]
            )

        # assert
        data = out.get()
        assert data.shape == grid
        focus = (-halo, -halo, -halo)
        for i in (
            range(halo, halo + grid[OUTER]) if cmn.n_dims > 1 else (INVALID_INDEX,)
        ):
            for j in (
                range(halo, halo + grid[MID3D]) if cmn.n_dims > 2 else (INVALID_INDEX,)
            ):
                for k in range(halo, halo + grid[INNER]):
                    if cmn.n_dims == 1:
                        ijk = (k, INVALID_INDEX, INVALID_INDEX)
                    elif cmn.n_dims == 2:
                        ijk = (i, k, INVALID_INDEX)
                    else:
                        ijk = (i, j, k)
                    value = cmn.traversals.indexers[cmn.n_dims].ats[
                        INNER if cmn.n_dims == 1 else OUTER
                    ](focus, data, *ijk)
                    assert (cmn.n_dims if loop else 1) * cell_id(i, j, k) == value
        assert cmn.scl_null_arg_impl[IMPL_META_AND_DATA][META_AND_DATA_META][
            META_HALO_VALID
        ]
        assert cmn.vec_null_arg_impl[IMPL_META_AND_DATA][META_AND_DATA_META][
            META_HALO_VALID
        ]
        assert not out.impl[IMPL_META_AND_DATA][META_AND_DATA_META][META_HALO_VALID]

    @staticmethod
    @pytest.mark.parametrize("halo", (1, 2, 3))
    @pytest.mark.parametrize("grid", ((3, 4, 5), (5, 6), (11,)))
    # pylint: disable-next=too-many-locals,redefined-outer-name
    def test_apply_vector(n_threads, halo: int, grid: tuple, left_first: bool = True):
        if len(grid) == 1 and n_threads > 1:
            return
        cmn = make_commons(grid, halo, n_threads, left_first)

        # arrange
        sut = cmn.traversals.apply_vector()

        data = {
            1: lambda: (np.zeros(grid[0] + 1),),
            2: lambda: (
                np.zeros((grid[0] + 1, grid[1])),
                np.zeros((grid[0], grid[1] + 1)),
            ),
            3: lambda: (
                np.zeros((grid[0] + 1, grid[1], grid[2])),
                np.zeros((grid[0], grid[1] + 1, grid[2])),
                np.zeros((grid[0], grid[1], grid[2] + 1)),
            ),
        }[cmn.n_dims]()

        out = VectorField(data, halo, tuple([Constant(np.nan)] * cmn.n_dims))
        out.assemble(cmn.traversals)

        # act
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NumbaExperimentalFeatureWarning)
            sut(
                *[_cell_id_vector] * MAX_DIM_NUM,
                *out.impl[IMPL_META_AND_DATA],
                *cmn.scl_null_arg_impl[IMPL_META_AND_DATA],
                *cmn.scl_null_arg_impl[IMPL_BC],
                *cmn.vec_null_arg_impl[IMPL_META_AND_DATA],
                *cmn.vec_null_arg_impl[IMPL_BC],
                *cmn.scl_null_arg_impl[IMPL_META_AND_DATA],
                *cmn.scl_null_arg_impl[IMPL_BC]
            )

        # assert
        dims = {1: (INNER,), 2: (OUTER, INNER), 3: (OUTER, MID3D, INNER)}[cmn.n_dims]

        for dim in dims:
            data = out.get_component(dim)
            focus = tuple(-cmn.halos[dim][i] for i in range(MAX_DIM_NUM))
            for i in (
                range(cmn.halos[dim][OUTER], cmn.halos[dim][OUTER] + data.shape[OUTER])
                if cmn.n_dims > 1
                else (INVALID_INDEX,)
            ):
                for j in (
                    range(
                        cmn.halos[dim][MID3D], cmn.halos[dim][MID3D] + data.shape[MID3D]
                    )
                    if cmn.n_dims > 2
                    else (INVALID_INDEX,)
                ):
                    for k in range(
                        cmn.halos[dim][INNER], cmn.halos[dim][INNER] + data.shape[INNER]
                    ):
                        if cmn.n_dims == 1:
                            ijk = (k, INVALID_INDEX, INVALID_INDEX)
                        elif cmn.n_dims == 2:
                            ijk = (i, k, INVALID_INDEX)
                        else:
                            ijk = (i, j, k)
                        value = cmn.traversals.indexers[cmn.n_dims].ats[
                            INNER if cmn.n_dims == 1 else OUTER
                        ](focus, data, *ijk)
                        assert cell_id(i, j, k) == value

        assert cmn.scl_null_arg_impl[IMPL_META_AND_DATA][META_AND_DATA_META][
            META_HALO_VALID
        ]
        assert cmn.vec_null_arg_impl[IMPL_META_AND_DATA][META_AND_DATA_META][
            META_HALO_VALID
        ]
        assert not out.impl[IMPL_META_AND_DATA][META_AND_DATA_META][META_HALO_VALID]

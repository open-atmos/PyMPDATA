# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
from functools import lru_cache
import pytest
import numba
import numpy as np
from PyMPDATA.impl.traversals import Traversals
from PyMPDATA.impl.meta import META_HALO_VALID
from PyMPDATA import Options, ScalarField, VectorField
from PyMPDATA.boundary_conditions import Constant
from PyMPDATA.impl.enumerations import (
    MAX_DIM_NUM, INNER, MID3D, OUTER, IMPL_META_AND_DATA, IMPL_BC,
    META_AND_DATA_META, ARG_FOCUS, INVALID_INDEX
)

# noinspection PyUnresolvedReferences
from tests.unit_tests.fixtures.n_threads import n_threads

jit_flags = Options().jit_flags


@lru_cache()
# pylint: disable-next=redefined-outer-name
def make_traversals(grid, halo, n_threads):
    return Traversals(grid=grid, halo=halo, jit_flags=jit_flags, n_threads=n_threads)


@numba.njit(**jit_flags)
def cell_id(i, j, k):
    if i == INVALID_INDEX:
        i = 0
    if j == INVALID_INDEX:
        j = 0
    return 100 * i + 10 * j + k


@numba.njit(**jit_flags)
def _cell_id_scalar(value, arg_1_vec, arg_2_scl, arg_3_scl, arg_4_scl, arg_5_scl):
    focus = arg_1_vec[ARG_FOCUS]
    if focus != arg_2_scl[ARG_FOCUS]:
        raise Exception()
    if focus != arg_3_scl[ARG_FOCUS]:
        raise Exception()
    if focus != arg_4_scl[ARG_FOCUS]:
        raise Exception()
    if focus != arg_5_scl[ARG_FOCUS]:
        raise Exception()
    return value + cell_id(*focus)


@numba.njit(**jit_flags)
def _cell_id_vector(arg_1, arg_2, arg_3):
    focus = arg_1[ARG_FOCUS]
    if focus != arg_2[ARG_FOCUS]:
        raise Exception()
    if focus != arg_3[ARG_FOCUS]:
        raise Exception()
    return cell_id(*focus)


class TestTraversals:
    @staticmethod
    @pytest.mark.parametrize("halo", (1, 2, 3))
    @pytest.mark.parametrize("grid", ((3, 4, 5), (5, 6), (11,)))
    @pytest.mark.parametrize("loop", (True, False))
    # pylint: disable-next=redefined-outer-name
    def test_apply_scalar(n_threads, halo, grid, loop):
        n_dims = len(grid)
        if n_dims == 1 and n_threads > 1:
            return

        # arrange
        traversals = make_traversals(grid, halo, n_threads)
        sut = traversals.apply_scalar(loop=loop)

        scl_null_arg_impl = ScalarField.make_null(n_dims, traversals).impl
        vec_null_arg_impl = VectorField.make_null(n_dims, traversals).impl

        out = ScalarField(np.zeros(grid), halo, [Constant(np.nan)] * n_dims)
        out.assemble(traversals)

        # act
        sut(_cell_id_scalar,
            _cell_id_scalar if loop else None,
            _cell_id_scalar if loop else None,
            *out.impl[IMPL_META_AND_DATA],
            *vec_null_arg_impl[IMPL_META_AND_DATA], *vec_null_arg_impl[IMPL_BC],
            *scl_null_arg_impl[IMPL_META_AND_DATA], *scl_null_arg_impl[IMPL_BC],
            *scl_null_arg_impl[IMPL_META_AND_DATA], *scl_null_arg_impl[IMPL_BC],
            *scl_null_arg_impl[IMPL_META_AND_DATA], *scl_null_arg_impl[IMPL_BC],
            *scl_null_arg_impl[IMPL_META_AND_DATA], *scl_null_arg_impl[IMPL_BC]
            )

        # assert
        data = out.get()
        assert data.shape == grid
        focus = (-halo, -halo, -halo)
        for i in range(halo, halo + grid[OUTER]) if n_dims > 1 else (INVALID_INDEX,):
            for j in range(halo, halo + grid[MID3D]) if n_dims > 2 else (INVALID_INDEX,):
                for k in range(halo, halo + grid[INNER]):
                    if n_dims == 1:
                        ijk = (k, INVALID_INDEX, INVALID_INDEX)
                    elif n_dims == 2:
                        ijk = (i, k, INVALID_INDEX)
                    else:
                        ijk = (i, j, k)
                    value = traversals.indexers[n_dims].ats[INNER if n_dims == 1 else OUTER](
                        focus, data, *ijk
                    )
                    assert (n_dims if loop else 1) * cell_id(i, j, k) == value
        assert scl_null_arg_impl[IMPL_META_AND_DATA][META_AND_DATA_META][META_HALO_VALID]
        assert vec_null_arg_impl[IMPL_META_AND_DATA][META_AND_DATA_META][META_HALO_VALID]
        assert not out.impl[IMPL_META_AND_DATA][META_AND_DATA_META][META_HALO_VALID]

    @staticmethod
    @pytest.mark.parametrize("halo", (1, 2, 3))
    @pytest.mark.parametrize("grid", ((3, 4, 5), (5, 6), (11,)))
    # pylint: disable-next=redefined-outer-name
    def test_apply_vector(n_threads, halo, grid):
        n_dims = len(grid)
        if n_dims == 1 and n_threads > 1:
            return

        # arrange
        traversals = make_traversals(grid, halo, n_threads)
        sut = traversals.apply_vector()

        scl_null_arg_impl = ScalarField.make_null(n_dims, traversals).impl
        vec_null_arg_impl = VectorField.make_null(n_dims, traversals).impl

        if n_dims == 1:
            data = (np.zeros(grid[0]+1),)
        elif n_dims == 2:
            data = (
                np.zeros((grid[0]+1, grid[1])),
                np.zeros((grid[0], grid[1]+1))
            )
        elif n_dims == 3:
            data = (
                np.zeros((grid[0]+1, grid[1], grid[2])),
                np.zeros((grid[0], grid[1]+1, grid[2])),
                np.zeros((grid[0], grid[1], grid[2]+1)),
            )
        else:
            raise NotImplementedError()

        out = VectorField(data, halo, [Constant(np.nan)] * n_dims)
        out.assemble(traversals)

        # act
        sut(*[_cell_id_vector] * MAX_DIM_NUM,
            *out.impl[IMPL_META_AND_DATA],
            *scl_null_arg_impl[IMPL_META_AND_DATA], *scl_null_arg_impl[IMPL_BC],
            *vec_null_arg_impl[IMPL_META_AND_DATA], *vec_null_arg_impl[IMPL_BC],
            *scl_null_arg_impl[IMPL_META_AND_DATA], *scl_null_arg_impl[IMPL_BC]
            )

        # assert
        halos = (
            (halo-1, halo, halo),
            (halo, halo-1, halo),
            (halo, halo, halo-1)
        )

        if n_dims == 1:
            dims = (INNER,)
        elif n_dims == 2:
            dims = (OUTER, INNER)
        else:
            dims = (OUTER, MID3D, INNER)
        for dim in dims:
            print("DIM", dim)
            data = out.get_component(dim)
            focus = tuple(-halos[dim][i] for i in range(MAX_DIM_NUM))
            print("focus", focus)
            for i in range(
                    halos[dim][OUTER],
                    halos[dim][OUTER] + data.shape[OUTER]
            ) if n_dims > 1 else (INVALID_INDEX,):
                for j in range(
                        halos[dim][MID3D],
                        halos[dim][MID3D] + data.shape[MID3D]
                ) if n_dims > 2 else (INVALID_INDEX,):
                    for k in range(halos[dim][INNER], halos[dim][INNER] + data.shape[INNER]):
                        if n_dims == 1:
                            ijk = (k, INVALID_INDEX, INVALID_INDEX)
                        elif n_dims == 2:
                            ijk = (i, k, INVALID_INDEX)
                        else:
                            ijk = (i, j, k)
                        print("check at", i, j, k)
                        value = traversals.indexers[n_dims].ats[INNER if n_dims == 1 else OUTER](
                            focus, data, *ijk
                        )
                        assert cell_id(i, j, k) == value

        assert scl_null_arg_impl[IMPL_META_AND_DATA][META_AND_DATA_META][META_HALO_VALID]
        assert vec_null_arg_impl[IMPL_META_AND_DATA][META_AND_DATA_META][META_HALO_VALID]
        assert not out.impl[IMPL_META_AND_DATA][META_AND_DATA_META][META_HALO_VALID]

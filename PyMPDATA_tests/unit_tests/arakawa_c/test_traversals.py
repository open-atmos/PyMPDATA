from PyMPDATA.arakawa_c.traversals import Traversals
from PyMPDATA.arakawa_c.meta import META_HALO_VALID
from PyMPDATA import Options, ScalarField, VectorField, ConstantBoundaryCondition
from PyMPDATA.arakawa_c.indexers import indexers
from PyMPDATA.arakawa_c.enumerations import MAX_DIM_NUM, INNER, MID3D, OUTER, IMPL_META_AND_DATA, IMPL_BC, \
    META_AND_DATA_META, ARG_FOCUS, INVALID_INDEX
import pytest
import numba
import numpy as np

jit_flags = Options().jit_flags
n_threads = (1, 2, 3)
try:
    numba.parfors.parfor.ensure_parallel_support()
except numba.core.errors.UnsupportedParforsError:
    n_threads = (1,)


@numba.njit(**jit_flags)
def cell_id(i, j, k):
    if i == INVALID_INDEX:
        i = 0
    if j == INVALID_INDEX:
        j = 0
    return 100 * i + 10 * j + k


@numba.njit(**jit_flags)
def _cell_id_scalar(value, arg_1_vec, arg_2_scl, arg_3_scl, arg_4_scl):
    focus = arg_1_vec[ARG_FOCUS]
    if focus != arg_2_scl[ARG_FOCUS]:
        raise Exception()
    if focus != arg_3_scl[ARG_FOCUS]:
        raise Exception()
    if focus != arg_4_scl[ARG_FOCUS]:
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
    @pytest.mark.parametrize("n_threads", n_threads)
    @pytest.mark.parametrize("halo", (1, 2, 3))
    @pytest.mark.parametrize("grid", ((5, 6), (11,)))  # TODO #96: 3d
    @pytest.mark.parametrize("loop", (True, False))
    def test_apply_scalar(n_threads, halo, grid, loop):
        n_dims = len(grid)
        if n_dims == 1 and n_threads > 1:
            return

        # arrange
        traversals = Traversals(grid, halo, jit_flags, n_threads)
        sut = traversals.apply_scalar(loop=loop)

        scl_null_arg_impl = ScalarField.make_null(n_dims).impl
        vec_null_arg_impl = VectorField.make_null(n_dims).impl

        out = ScalarField(np.zeros(grid), halo, [ConstantBoundaryCondition(np.nan)]*n_dims)

        # act
        sut(_cell_id_scalar,
            _cell_id_scalar if loop else None,
            _cell_id_scalar if loop else None,
            *out.impl[IMPL_META_AND_DATA],
            *vec_null_arg_impl[IMPL_META_AND_DATA], *vec_null_arg_impl[IMPL_BC],
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
                        raise NotImplementedError()
                    value = indexers[n_dims].at[INNER if n_dims == 1 else OUTER](focus, data, *ijk)
                    assert (n_dims if loop else 1) * cell_id(i, j, k) == value
        assert scl_null_arg_impl[IMPL_META_AND_DATA][META_AND_DATA_META][META_HALO_VALID]
        assert vec_null_arg_impl[IMPL_META_AND_DATA][META_AND_DATA_META][META_HALO_VALID]
        assert not out.impl[IMPL_META_AND_DATA][META_AND_DATA_META][META_HALO_VALID]

    @staticmethod
    @pytest.mark.parametrize("n_threads", n_threads)
    @pytest.mark.parametrize("halo", (1, 2, 3))
    @pytest.mark.parametrize("grid", ((5, 6), (11,)))  # TODO #96 - 3d
    def test_apply_vector(n_threads, halo, grid):
        n_dims = len(grid)
        if n_dims == 1 and n_threads > 1:
            return

        # arrange
        traversals = Traversals(grid, halo, jit_flags, n_threads)
        sut = traversals.apply_vector()

        scl_null_arg_impl = ScalarField.make_null(n_dims).impl
        vec_null_arg_impl = VectorField.make_null(n_dims).impl

        if n_dims == 1:
            data = (np.zeros(grid[0]+1),)
        elif n_dims == 2:
            data = (
                np.zeros((grid[0]+1, grid[1])),
                np.zeros((grid[0], grid[1]+1))
            )
        elif n_dims == 3:
            pass  # TODO #96
        else:
            raise NotImplementedError()

        out = VectorField(data, halo, [ConstantBoundaryCondition(np.nan)] * n_dims)

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
            raise NotImplementedError()
        for d in dims:
            print("DIM", d)
            data = out.get_component(d)
            focus = tuple(-halos[d][i] for i in range(MAX_DIM_NUM))
            print("focus", focus)
            for i in range(halos[d][OUTER], halos[d][OUTER] + data.shape[OUTER]) if n_dims > 1 else (INVALID_INDEX,):
                for j in range(halos[d][MID3D], halos[d][MID3D] + data.shape[MID3D]) if n_dims > 2 else (INVALID_INDEX,):
                    for k in range(halos[d][INNER], halos[d][INNER] + data.shape[INNER]):
                        if n_dims == 1:
                            ijk = (k, INVALID_INDEX, INVALID_INDEX)
                        elif n_dims == 2:
                            ijk = (i, k, INVALID_INDEX)
                        else:
                            raise NotImplementedError()
                        print("check at", i, j, k)
                        value = indexers[n_dims].at[INNER if n_dims == 1 else OUTER](focus, data, *ijk)
                        assert cell_id(i, j, k) == value

        assert scl_null_arg_impl[IMPL_META_AND_DATA][META_AND_DATA_META][META_HALO_VALID]
        assert vec_null_arg_impl[IMPL_META_AND_DATA][META_AND_DATA_META][META_HALO_VALID]
        assert not out.impl[IMPL_META_AND_DATA][META_AND_DATA_META][META_HALO_VALID]

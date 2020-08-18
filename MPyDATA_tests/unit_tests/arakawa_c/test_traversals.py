from MPyDATA.arakawa_c.traversals import Traversals
from MPyDATA.arakawa_c.meta import META_HALO_VALID
from MPyDATA import Options, ScalarField, VectorField, ConstantBoundaryCondition
from MPyDATA.arakawa_c.indexers import indexers
from MPyDATA.arakawa_c.enumerations import MAX_DIM_NUM, INNER, OUTER, IMPL_META_AND_DATA, IMPL_BC, META_AND_DATA_META, ARG_FOCUS
import pytest
import numba
import numpy as np

jit_flags = Options().jit_flags


@numba.njit(**jit_flags)
def cell_id(i, j):
    if i == -1:
        return j
    return 100 * i + j


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
    @pytest.mark.parametrize("n_threads", (1, 2, 3))
    @pytest.mark.parametrize("halo", (1, 2, 3))
    @pytest.mark.parametrize("grid", ((5, 6), (11,)))
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
        sut(_cell_id_scalar, _cell_id_scalar,
            *out.impl[IMPL_META_AND_DATA],
            *vec_null_arg_impl[IMPL_META_AND_DATA], *vec_null_arg_impl[IMPL_BC],
            *scl_null_arg_impl[IMPL_META_AND_DATA], *scl_null_arg_impl[IMPL_BC],
            *scl_null_arg_impl[IMPL_META_AND_DATA], *scl_null_arg_impl[IMPL_BC],
            *scl_null_arg_impl[IMPL_META_AND_DATA], *scl_null_arg_impl[IMPL_BC]
            )

        # assert
        data = out.get()
        assert data.shape == grid
        focus = (-halo, -halo)
        for i in (-1,) if n_dims == 1 else range(halo, halo + grid[0]):
            for j in range(halo, halo + grid[-1]):
                ij = (i, j) if n_dims == 2 else (j, i)
                value = indexers[n_dims].at[MAX_DIM_NUM-n_dims](focus, data, *ij)
                assert (n_dims if loop else 1) * cell_id(i, j) == value
        assert scl_null_arg_impl[IMPL_META_AND_DATA][META_AND_DATA_META][META_HALO_VALID]
        assert vec_null_arg_impl[IMPL_META_AND_DATA][META_AND_DATA_META][META_HALO_VALID]
        assert not out.impl[IMPL_META_AND_DATA][META_AND_DATA_META][META_HALO_VALID]

    @staticmethod
    @pytest.mark.parametrize("n_threads", (1, 2, 3))
    @pytest.mark.parametrize("halo", (1, 2, 3))
    @pytest.mark.parametrize("grid", ((5, 6), (11,)))
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
        else:
            raise NotImplementedError()

        out = VectorField(data, halo, [ConstantBoundaryCondition(np.nan)] * n_dims)

        # act
        sut(_cell_id_vector, _cell_id_vector,
            *out.impl[IMPL_META_AND_DATA],
            *scl_null_arg_impl[IMPL_META_AND_DATA], *scl_null_arg_impl[IMPL_BC],
            *vec_null_arg_impl[IMPL_META_AND_DATA], *vec_null_arg_impl[IMPL_BC],
            *scl_null_arg_impl[IMPL_META_AND_DATA], *scl_null_arg_impl[IMPL_BC]
            )

        # assert
        if n_dims == 1:
            halos = ((-1, halo-1),)
        elif n_dims == 2:
            halos = (
                (halo-1, halo),
                (halo, halo-1)
            )
        else:
            raise NotImplementedError()
        for d in range(n_dims):
            data = out.get_component(d)
            focus = tuple(-halos[d][i] for i in range(MAX_DIM_NUM))
            for i in (-1,) if n_dims == 1 else range(halos[d][OUTER], halos[d][OUTER] + data.shape[0]):
                for j in range(halos[d][INNER], halos[d][INNER] + data.shape[-1]):
                    ij = (i, j) if n_dims == 2 else (j, i)
                    value = indexers[n_dims].at[MAX_DIM_NUM-n_dims](focus, data, *ij)
                    assert cell_id(i, j) == value

        assert scl_null_arg_impl[IMPL_META_AND_DATA][META_AND_DATA_META][META_HALO_VALID]
        assert vec_null_arg_impl[IMPL_META_AND_DATA][META_AND_DATA_META][META_HALO_VALID]
        assert not out.impl[IMPL_META_AND_DATA][META_AND_DATA_META][META_HALO_VALID]

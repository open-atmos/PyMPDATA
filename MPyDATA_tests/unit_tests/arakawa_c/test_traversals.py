from MPyDATA.arakawa_c.traversals import Traversals
from MPyDATA.arakawa_c.meta import meta_halo_valid
from MPyDATA import Options, ScalarField, VectorField, ConstantBoundaryCondition
import pytest
import numba
import numpy as np

jit_flags = Options().jit_flags


@numba.njit(**jit_flags)
def _increment_scalar(value, _1, _2, _3, _4):
    return value + 1


@numba.njit(**jit_flags)
def _ones_vector(_1, _2, _3):
    return 1


class TestTraversals:
    @staticmethod
    @pytest.mark.parametrize("n_threads", (1, 2, 3))
    @pytest.mark.parametrize("halo", (1, 2, 3))
    @pytest.mark.parametrize("grid", ((5, 6), (11,)))
    @pytest.mark.parametrize("loop", (True, False))
    def test_apply_scalar(n_threads, halo, grid, loop):
        if len(grid) == 1 and n_threads > 1:
            return

        # arrange
        traversals = Traversals(grid, halo, jit_flags, n_threads)
        sut = traversals.apply_scalar(loop=loop)

        scl_null_arg_impl = ScalarField.make_null(len(grid)).impl
        vec_null_arg_impl = VectorField.make_null(len(grid)).impl

        out = ScalarField(np.zeros(grid), halo, [ConstantBoundaryCondition(np.nan)]*len(grid))

        # act
        sut(_increment_scalar, _increment_scalar,
            *out.impl[0],
            *vec_null_arg_impl[0], *vec_null_arg_impl[1],
            *scl_null_arg_impl[0], *scl_null_arg_impl[1],
            *scl_null_arg_impl[0], *scl_null_arg_impl[1],
            *scl_null_arg_impl[0], *scl_null_arg_impl[1]
            )

        # assert
        assert (out.get() == (len(grid) if loop else 1)).all()
        assert scl_null_arg_impl[0][0][meta_halo_valid]
        assert vec_null_arg_impl[0][0][meta_halo_valid]
        assert not out.impl[0][0][meta_halo_valid]

    @staticmethod
    @pytest.mark.parametrize("n_threads", (1, 2, 3))
    @pytest.mark.parametrize("halo", (1, 2, 3))
    @pytest.mark.parametrize("grid", ((5, 6), (11,)))
    def test_apply_vector(n_threads, halo, grid):
        if len(grid) == 1 and n_threads > 1:
            return

        # arrange
        traversals = Traversals(grid, halo, jit_flags, n_threads)
        sut = traversals.apply_vector()

        scl_null_arg_impl = ScalarField.make_null(len(grid)).impl
        vec_null_arg_impl = VectorField.make_null(len(grid)).impl

        n_dims = len(grid)
        if n_dims == 1:
            data = (np.zeros(grid[0]+1),)
        elif n_dims == 2:
            data = (
                np.zeros((grid[0]+1, grid[1])),
                np.zeros((grid[0], grid[1]+1))
            )
        else:
            raise NotImplementedError()

        out = VectorField(data, halo, [ConstantBoundaryCondition(np.nan)]*len(grid))

        # act
        sut(_ones_vector, _ones_vector,
            *out.impl[0],
            *scl_null_arg_impl[0], *scl_null_arg_impl[1],
            *vec_null_arg_impl[0], *vec_null_arg_impl[1],
            *scl_null_arg_impl[0], *scl_null_arg_impl[1]
            )

        # assert
        for i in range(n_dims):
            assert (out.get_component(i) == 1).all()
        assert scl_null_arg_impl[0][0][meta_halo_valid]
        assert vec_null_arg_impl[0][0][meta_halo_valid]
        assert not out.impl[0][0][meta_halo_valid]

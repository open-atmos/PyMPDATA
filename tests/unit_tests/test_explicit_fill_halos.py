# pylint:disable=missing-module-docstring,missing-function-docstring
import numpy as np
import pytest

from PyMPDATA import Options, ScalarField, VectorField
from PyMPDATA.boundary_conditions import Periodic
from PyMPDATA.impl.enumerations import MAX_DIM_NUM
from PyMPDATA.impl.traversals import Traversals

JIT_FLAGS = Options().jit_flags


def assert_slice_size(s: slice, halo):
    if s.stop is None:
        if abs(s.start) != halo and abs(s.start) != halo - 1:
            raise AssertionError("Slice and halo size mismatch")
    elif s.stop is not None:
        if abs(s.stop - s.start) != halo and abs(s.stop - s.start) != halo - 1:
            raise AssertionError("Slice and halo size mismatch")
    else:
        assert False


def assert_array_not_equal(a, b):
    return np.testing.assert_raises(AssertionError, np.testing.assert_array_equal, a, b)


@pytest.mark.parametrize("bc", (Periodic(),))
@pytest.mark.parametrize("n_threads", (1,))
@pytest.mark.parametrize("halo", (1, 2, 3))
@pytest.mark.parametrize(
    "field_factory",
    (
        lambda halo, bc: ScalarField(np.zeros(3), halo, bc),  # 1d
        lambda halo, bc: VectorField((np.zeros(3),), halo, bc),  # 1d
        lambda halo, bc: ScalarField(np.zeros((3, 3)), halo, bc),  # 2d
        lambda halo, bc: VectorField(
            (
                np.zeros(
                    (4, 3),
                ),
                np.zeros(
                    (3, 4),
                ),
            ),
            halo,
            bc,
        ),  # 2d
    ),
)
def test_explicit_fill_halos(field_factory, halo, bc, n_threads):
    # arange
    field = field_factory(halo, (bc, bc))
    traversals = Traversals(
        grid=field.grid,
        halo=halo,
        jit_flags=JIT_FLAGS,
        n_threads=n_threads,
        left_first=tuple([True] * MAX_DIM_NUM),
        buffer_size=0,
    )
    field.assemble(traversals)
    if isinstance(field, ScalarField):
        field.get()[:] = np.arange(1, field.grid[0] + 1)
        left_halo = slice(0, halo)
        right_halo = slice(-halo, None)
        left_edge = slice(halo, 2 * halo)
        right_edge = slice(-2 * halo, -halo)
        slices = [left_halo, right_halo, left_edge, right_edge]
        for s in slices:
            assert_slice_size(s, halo)
        data = field.data
    elif isinstance(field, VectorField):
        if field.get_component(0)[:].ndim > 1:
            field.get_component(0)[0][:] = np.arange(1, field.grid[0] + 1)
            field.get_component(0)[1][:] = np.arange(1, field.grid[0] + 1)
        else:
            field.get_component(0)[:] = np.arange(1, field.grid[0] + 2)
        if halo == 1:
            pytest.skip()
        left_halo = slice(0, halo - 1)
        right_halo = slice(-(halo - 1), None)
        left_edge = slice(halo, 2 * (halo - 1) + 1)
        right_edge = slice(-2 * (halo - 1) - 1, -(halo - 1) - 1)
        slices = [left_halo, right_halo, left_edge, right_edge]
        for s in slices:
            assert_slice_size(s, halo)
        data = field.data[0]
    else:
        assert False
    assert_array_not_equal(data[left_halo], data[right_edge])
    assert_array_not_equal(data[right_halo], data[left_edge])

    # act
    # pylint:disable=protected-access
    field._debug_fill_halos(traversals, range(n_threads))

    # assert
    np.testing.assert_array_equal(data[left_halo], data[right_edge], verbose=True)
    np.testing.assert_array_equal(data[right_halo], data[left_edge], verbose=True)

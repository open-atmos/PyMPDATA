import numpy as np
import pytest

from PyMPDATA import Options, ScalarField, VectorField
from PyMPDATA.boundary_conditions import Periodic
from PyMPDATA.impl.enumerations import MAX_DIM_NUM
from PyMPDATA.impl.traversals import Traversals

JIT_FLAGS = Options().jit_flags


@pytest.mark.parametrize("bc", (Periodic(),))
@pytest.mark.parametrize("n_threads", (1,))
@pytest.mark.parametrize("halo", (1, 2, 3))
@pytest.mark.parametrize(
    "field_factory",
    (
        lambda halo, bc: ScalarField(np.zeros(3), halo, bc),  # 1d
        lambda halo, bc: VectorField((np.zeros(3),), halo, bc),  # 1d
        # ScalarField, #2d
        # VectorField, #2d
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
    # TODO: Assert all slices are of halo or halo-1 size
    if isinstance(field, ScalarField):
        field.get()[:] = np.arange(1, field.grid[0] + 1)
        left_halo = slice(0, halo)
        right_halo = slice(-halo, None)
        left_edge = slice(halo, 2 * halo)
        right_edge = slice(-2 * halo, -halo)
        data = field.data
    elif isinstance(field, VectorField):
        field.get_component(0)[:] = np.arange(1, field.grid[0] + 2)
        if halo == 1:
            pytest.skip()
        left_halo = slice(0, halo - 1)
        right_halo = slice(-(halo - 1), None)
        left_edge = slice(halo, 2 * (halo - 1) + 1)
        right_edge = slice(-2 * (halo - 1) - 1, -(halo - 1) - 1)
        data = field.data[0]
    else:
        assert False
    assert all(data[left_halo] != data[right_edge])
    assert all(data[right_halo] != data[left_edge])

    # act
    field._debug_fill_halos(traversals, range(n_threads))  # pylint:disable=protected...

    # assert
    assert all(data[left_halo] == data[right_edge])
    assert all(data[right_halo] == data[left_edge])

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring,invalid-name
import numpy as np
import pytest

from PyMPDATA import Options, ScalarField, VectorField
from PyMPDATA.boundary_conditions import Constant
from PyMPDATA.impl import indexers
from PyMPDATA.impl.enumerations import INNER, OUTER

options = Options()
bc = [Constant(value=0)]
indexers = indexers.make_indexers(options.jit_flags)
halo = options.n_halo
focus = (0, 0, 0)

rng = np.random.default_rng()


def test_ats1D():
    # arrange
    grid = 5
    inp_arr = rng.random(grid)
    field = ScalarField(inp_arr, halo, bc)
    arr = field.data
    sut = indexers[1].ats[INNER]
    index = 2

    # act
    value = sut(focus, arr, index)

    # assert
    assert value == arr[index]


def test_atv1D():
    # arrange
    grid = 5
    field = VectorField((rng.random(size=grid),), halo, bc)
    arrs = field.data
    sut = indexers[1].atv[INNER]
    index_vec = 2.5
    index = 2

    # act
    value = sut(focus, arrs, index_vec)

    # assert
    assert value == arrs[INNER][index]


def test_ati1D():
    # arrange
    grid = 5
    field = VectorField((rng.random(size=grid),), halo, bc)
    arrs = field.data
    sut = indexers[1].ati[INNER]
    index_vec = 2.5
    index = 2

    # act
    value = sut(focus, arrs, index_vec)

    # assert
    assert value == (arrs[INNER][index] + arrs[INNER][index + 1]) / 2


@pytest.mark.parametrize("axis", (INNER, OUTER))
def test_ats2D(axis):
    # arrange
    grid = (5, 5)
    bc_2D = [bc] * len(grid)
    field = ScalarField(rng.random(grid), halo, bc_2D)
    arr = field.data
    sut = indexers[2].ats[axis]
    index = 2

    # act
    value = sut(focus, arr, index, halo)

    # assert
    assert value == arr[((index, halo), (halo, index))[axis]]


@pytest.mark.parametrize("axis", (INNER, OUTER))
def test_atv2D(axis):
    # arrange
    xi = 5
    yi = 5
    bc_2D = [bc] * len((xi, yi))
    field = VectorField(
        (rng.random((xi + 1, yi)), rng.random((xi, yi + 1))), options.n_halo, bc_2D
    )
    arrs = field.data
    sut = indexers[2].atv[axis]
    index_vec = 2.5
    index = 2

    # act
    value = sut(focus, arrs, index_vec, halo)

    # assert
    assert value == arrs[axis][((index, halo), (halo, index))[axis]]


@pytest.mark.parametrize("axis", (INNER, OUTER))
def test_ati2D(axis):
    # arrange
    grid = 5, 5
    bc_2D = [bc] * len((grid))
    fields = (
        ScalarField(rng.random(grid), halo, bc_2D),
        ScalarField(rng.random(grid), halo, bc_2D),
    )
    arrs = (fields[INNER].data, fields[OUTER].data)
    sut = indexers[2].ati[axis]
    index_vec = 2.5
    index = 2

    # act
    value = sut(focus, arrs, index_vec, halo)

    # assert
    assert (
        value
        == (
            arrs[axis][((index, halo), (halo, index))[axis]]
            + arrs[axis][((index + 1, halo), (halo, index + 1))[axis]]
        )
        / 2
    )

# pylint: disable=missing-module-docstring,missing-function-docstring
import warnings

import numpy as np
from numba.core.errors import NumbaExperimentalFeatureWarning

from PyMPDATA import Options, ScalarField
from PyMPDATA.boundary_conditions import Constant
from PyMPDATA.impl.enumerations import (
    ARG_DATA,
    IMPL_META_AND_DATA,
    INNER,
    MAX_DIM_NUM,
    META_AND_DATA_META,
    MID3D,
    OUTER,
)
from PyMPDATA.impl.formulae_divide import make_divide_or_zero
from PyMPDATA.impl.meta import META_HALO_VALID
from PyMPDATA.impl.traversals import Traversals


def test_divide():
    # Arrange
    np.random.seed(1)
    time_step = 1
    grid_step = (1.25, 0.75, 0.25)

    data_divisor = np.ones((4, 4, 3), dtype=float) * 2

    data = tuple(np.random.uniform(0, 10, (4, 4, 3)) for dim in range(3))
    expected = tuple(
        np.divide(data[dim], data_divisor) * time_step / grid_step[dim]
        for dim in range(3)
    )

    options = Options()
    halo = options.n_halo
    traversals = Traversals(
        buffer_size=0,
        grid=data[INNER].shape,
        halo=halo,
        jit_flags=options.jit_flags,
        n_threads=1,
        left_first=tuple([True] * MAX_DIM_NUM),
    )
    divide_or_zero = make_divide_or_zero(options, traversals)

    boundary_condition = tuple([Constant(value=0)] * 3)

    inputs = tuple(ScalarField(data[dim], halo, boundary_condition) for dim in range(3))
    for field in inputs:
        field.assemble(traversals)
    inputs = tuple(field.impl for field in inputs)

    outputs = tuple(
        ScalarField(np.zeros_like(data[dim]), halo, boundary_condition)
        for dim in range(3)
    )
    for field in outputs:
        field.assemble(traversals)
    outputs = tuple(field.impl for field in outputs)

    divisor = ScalarField(data_divisor, halo, boundary_condition)
    divisor.assemble(traversals)

    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NumbaExperimentalFeatureWarning)
        divide_or_zero(
            *outputs[OUTER][IMPL_META_AND_DATA],
            *outputs[MID3D][IMPL_META_AND_DATA],
            *outputs[INNER][IMPL_META_AND_DATA],
            *inputs[OUTER][IMPL_META_AND_DATA],
            *inputs[MID3D][IMPL_META_AND_DATA],
            *inputs[INNER][IMPL_META_AND_DATA],
            traversals.data.buffer,
            divisor.impl[IMPL_META_AND_DATA][ARG_DATA],
            time_step,
            grid_step
        )

    # Assert
    for axis in range(3):
        assert np.isfinite(
            outputs[axis][IMPL_META_AND_DATA][ARG_DATA][
                halo:-halo, halo:-halo, halo:-halo
            ]
        ).all()
    assert not outputs[axis][IMPL_META_AND_DATA][META_AND_DATA_META][META_HALO_VALID]
    np.testing.assert_array_equal(
        outputs[axis][IMPL_META_AND_DATA][ARG_DATA][halo:-halo, halo:-halo, halo:-halo],
        expected[axis],
    )

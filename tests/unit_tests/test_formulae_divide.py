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


# pylint: disable=too-many-locals
def test_divide():
    # Arrange
    np.random.seed(1)
    time_step = 1
    grid_step = (1.25, 0.75, 0.25)

    data_divisor = np.ones((4, 4, 3), dtype=float) * 2

    data_outer = np.random.uniform(0, 10, (4, 4, 3))
    data_mid3d = np.random.uniform(0, 10, (4, 4, 3))
    data_inner = np.random.uniform(0, 10, (4, 4, 3))

    expected_outer = np.divide(data_outer, data_divisor) * time_step / grid_step[OUTER]
    expected_mid3d = np.divide(data_mid3d, data_divisor) * time_step / grid_step[MID3D]
    expected_inner = np.divide(data_inner, data_divisor) * time_step / grid_step[INNER]

    expected = [expected_outer, expected_mid3d, expected_inner]

    options = Options()
    halo = options.n_halo
    # pylint: disable=duplicate-code
    traversals = Traversals(
        grid=data_inner.shape,
        halo=halo,
        jit_flags=options.jit_flags,
        n_threads=1,
        left_first=tuple([True] * MAX_DIM_NUM),
        buffer_size=0,
    )
    divide_or_zero = make_divide_or_zero(options, traversals)

    boundary_condition = [
        Constant(value=0),
    ] * 3

    input_outer = ScalarField(data_outer, halo, boundary_condition)
    input_outer.assemble(traversals)
    input_outer_impl = input_outer.impl

    input_mid3d = ScalarField(data_mid3d, halo, boundary_condition)
    input_mid3d.assemble(traversals)
    input_mid3d_impl = input_mid3d.impl

    input_inner = ScalarField(data_inner, halo, boundary_condition)
    input_inner.assemble(traversals)
    input_inner_impl = input_inner.impl

    output_outer = ScalarField(np.zeros_like(data_outer), halo, boundary_condition)
    output_outer.assemble(traversals)
    output_outer_impl = output_outer.impl

    output_mid3d = ScalarField(np.zeros_like(data_mid3d), halo, boundary_condition)
    output_mid3d.assemble(traversals)
    output_mid3d_impl = output_mid3d.impl

    output_inner = ScalarField(np.zeros_like(data_inner), halo, boundary_condition)
    output_inner.assemble(traversals)
    output_inner_impl = output_inner.impl

    outputs = [output_outer_impl, output_mid3d_impl, output_inner_impl]

    divisor = ScalarField(data_divisor, halo, boundary_condition)
    divisor.assemble(traversals)
    divisor_impl = divisor.impl

    traversals_data = traversals.data

    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NumbaExperimentalFeatureWarning)
        divide_or_zero(
            *output_outer_impl[IMPL_META_AND_DATA],
            *output_mid3d_impl[IMPL_META_AND_DATA],
            *output_inner_impl[IMPL_META_AND_DATA],
            *input_outer_impl[IMPL_META_AND_DATA],
            *input_mid3d_impl[IMPL_META_AND_DATA],
            *input_inner_impl[IMPL_META_AND_DATA],
            traversals_data.buffer,
            divisor_impl[IMPL_META_AND_DATA][ARG_DATA],
            time_step,
            grid_step
        )

    # Assert
    for axis in [OUTER, MID3D, INNER]:
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

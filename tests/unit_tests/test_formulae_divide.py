# pylint: disable=missing-module-docstring,missing-function-docstring
import warnings

import numpy as np
from numba.core.errors import NumbaExperimentalFeatureWarning

from PyMPDATA import Options, ScalarField
from PyMPDATA.boundary_conditions import Constant
from PyMPDATA.impl.enumerations import (
    ARG_DATA,
    IMPL_META_AND_DATA,
    MAX_DIM_NUM,
    META_AND_DATA_META,
)
from PyMPDATA.impl.formulae_divide import make_divide_or_zero
from PyMPDATA.impl.meta import META_HALO_VALID
from PyMPDATA.impl.traversals import Traversals


# pylint: disable=too-many-locals
def test_divide():
    # Arrange
    data_input = np.array((2, 4, 6, 8), dtype=float)
    data_divisor = np.array((2, 2, 2, 2), dtype=float)
    expected = np.array((4, 8, 12, 16), dtype=float)

    options = Options()
    halo = options.n_halo
    # pylint: disable=duplicate-code
    traversals = Traversals(
        grid=data_input.shape,
        halo=halo,
        jit_flags=options.jit_flags,
        n_threads=1,
        left_first=tuple([True] * MAX_DIM_NUM),
        buffer_size=0,
    )
    divide_or_zero = make_divide_or_zero(options, traversals)

    boundary_condition = (Constant(value=0),)

    input_field = ScalarField(data_input, halo, boundary_condition)
    input_field.assemble(traversals)
    input_field_impl = input_field.impl

    output_field = ScalarField(np.zeros_like(data_input), halo, boundary_condition)
    output_field.assemble(traversals)
    output_field_impl = output_field.impl

    divisor = ScalarField(data_divisor, halo, boundary_condition)
    divisor.assemble(traversals)
    divisor_impl = divisor.impl

    time_step = 1
    grid_step = (None, None, 0.25)
    traversals_data = traversals.data

    # Act
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=NumbaExperimentalFeatureWarning)
        divide_or_zero(
            None,
            None,
            None,
            None,
            *output_field_impl[IMPL_META_AND_DATA],
            None,
            None,
            None,
            None,
            *input_field_impl[IMPL_META_AND_DATA],
            traversals_data.buffer,
            divisor_impl[IMPL_META_AND_DATA][ARG_DATA],
            time_step,
            grid_step
        )

    # Assert
    assert np.isfinite(
        output_field_impl[IMPL_META_AND_DATA][ARG_DATA][halo:-halo]
    ).all()
    assert not output_field_impl[IMPL_META_AND_DATA][META_AND_DATA_META][
        META_HALO_VALID
    ]
    np.testing.assert_array_equal(
        output_field_impl[IMPL_META_AND_DATA][ARG_DATA][halo:-halo], expected
    )

"""Operation logic for dividing the field by
a set divisor table and saving the result
to a seperate field. Requires 'dynmaic_advector'
option to be enabled.
Scalar field inputs are named after dimensional
components of the VectorField, not to be confused
with internal enumerations on axis indexing"""

import numba
import numpy as np

from .enumerations import INNER, MID3D, OUTER
from .meta import META_HALO_VALID


def make_divide_or_zero(options, traversals):
    """returns njit-ted function for use with given traversals"""

    n_dims = traversals.n_dims

    @numba.njit(**options.jit_flags)
    # pylint: disable=too-many-arguments
    def divide_or_zero(
        out_outer_meta,
        out_outer_data,
        out_mid3d_meta,
        out_mid3d_data,
        out_inner_meta,
        out_inner_data,
        _,
        dividend_outer,
        __,
        dividend_mid3d,
        ___,
        dividend_inner,
        ____,
        divisor,
        time_step,
        grid_step,
    ):
        eps = 1e-7
        for i in np.ndindex(out_inner_data.shape):
            if n_dims > 1:
                out_outer_data[i] = (
                    dividend_outer[i] / divisor[i] * time_step / grid_step[OUTER]
                    if divisor[i] > eps
                    else 0
                )
                if n_dims > 2:
                    out_mid3d_data[i] = (
                        dividend_mid3d[i] / divisor[i] * time_step / grid_step[MID3D]
                        if divisor[i] > eps
                        else 0
                    )
            out_inner_data[i] = (
                dividend_inner[i] / divisor[i] * time_step / grid_step[INNER]
                if divisor[i] > eps
                else 0
            )
        if n_dims > 1:
            out_outer_meta[META_HALO_VALID] = False
            if n_dims > 2:
                out_mid3d_meta[META_HALO_VALID] = False
        out_inner_meta[META_HALO_VALID] = False

    return divide_or_zero

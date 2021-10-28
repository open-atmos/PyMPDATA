""" basic a*x+y operation logic for use in Fickian term handling """
import numba
from .enumerations import OUTER, MID3D, INNER
from .meta import META_HALO_VALID


def make_axpy(options, traversals):
    """ returns njit-ted function for use with given traversals """

    n_dims = traversals.n_dims

    @numba.njit(**options.jit_flags)
    # pylint: disable=too-many-arguments
    def axpy(out_meta, out_outer, out_mid3d, out_inner, a_coeffs,
             _, x_outer, x_mid3d, x_inner,
             __, y_outer, y_mid3d, y_inner):
        if n_dims > 1:
            out_outer[:] = a_coeffs[OUTER] * x_outer[:] + y_outer[:]
            if n_dims > 2:
                out_mid3d[:] = a_coeffs[MID3D] * x_mid3d[:] + y_mid3d[:]
        out_inner[:] = a_coeffs[INNER] * x_inner[:] + y_inner[:]
        out_meta[META_HALO_VALID] = False
    return axpy

""" array indexing logic for 1D, 2D and 3D staggered grids """
# pylint: disable=missing-function-docstring

from collections import namedtuple
from pathlib import Path

import numba

from .enumerations import INNER, INVALID_INDEX, MID3D, OUTER


def make_indexers(jit_flags):
    """returns a tuple indexed by dimension (0: None, 1: 1D, ...)
    with each element set to a namedtuple with 'at', 'atv', 'set' and 'get' functions"""

    @numba.njit([numba.boolean(numba.float64), numba.boolean(numba.int64)], **jit_flags)
    def _is_integral(value):
        return int(value * 2.0) % 2 == 0

    class _1D:
        @staticmethod
        @numba.njit(**jit_flags)
        def ats_1d(focus, arr, k, _=INVALID_INDEX, __=INVALID_INDEX):
            return arr[focus[INNER] + k]

        @staticmethod
        @numba.njit(**jit_flags)
        def atv_1d(focus, arrs, k, _=INVALID_INDEX, __=INVALID_INDEX):
            return arrs[INNER][focus[INNER] + int(k - 0.5)]

        @staticmethod
        @numba.njit(**jit_flags)
        def set(arr, _, __, k, value):
            arr[k] = value

        @staticmethod
        @numba.njit(**jit_flags)
        def get(arr, _, __, k):
            return arr[k]

    class _2D:
        @staticmethod
        @numba.njit(**jit_flags)
        def ats_axis0(focus, arr, i, k=0, _=INVALID_INDEX):
            return arr[focus[OUTER] + i, focus[INNER] + k]

        @staticmethod
        @numba.njit(**jit_flags)
        def ats_axis1(focus, arr, k, i=0, _=INVALID_INDEX):
            return arr[focus[OUTER] + i, focus[INNER] + k]

        @staticmethod
        @numba.njit(**jit_flags)
        def atv_axis0(focus, arrs, i, k=0, _=INVALID_INDEX):
            if _is_integral(i):
                dim, _ii, _kk = INNER, int(i), int(k - 0.5)
            else:
                dim, _ii, _kk = OUTER, int(i - 0.5), int(k)
            return arrs[dim][focus[OUTER] + _ii, focus[INNER] + _kk]

        @staticmethod
        @numba.njit(**jit_flags)
        def atv_axis1(focus, arrs, k, i=0, _=INVALID_INDEX):
            if _is_integral(i):
                dim, _ii, _kk = INNER, int(i), int(k - 0.5)
            else:
                dim, _ii, _kk = OUTER, int(i - 0.5), int(k)
            return arrs[dim][focus[OUTER] + _ii, focus[INNER] + _kk]

        @staticmethod
        @numba.njit(**jit_flags)
        def set(arr, i, _, k, value):
            arr[i, k] = value

        @staticmethod
        @numba.njit(**jit_flags)
        def get(arr, i, _, k):
            return arr[i, k]

    class _3D:
        @staticmethod
        @numba.njit(**jit_flags)
        def ats_axis0(focus, arr, i, j=0, k=0):
            return arr[focus[OUTER] + i, focus[MID3D] + j, focus[INNER] + k]

        @staticmethod
        @numba.njit(**jit_flags)
        def ats_axis1(focus, arr, j, k=0, i=0):
            return arr[focus[OUTER] + i, focus[MID3D] + j, focus[INNER] + k]

        @staticmethod
        @numba.njit(**jit_flags)
        def ats_axis2(focus, arr, k, i=0, j=0):
            return arr[focus[OUTER] + i, focus[MID3D] + j, focus[INNER] + k]

        @staticmethod
        @numba.njit(**jit_flags)
        def atv_axis0(focus, arrs, i, j=0, k=0):
            if not _is_integral(i):
                dim, _ii, _jj, _kk = OUTER, int(i - 0.5), int(j), int(k)
            elif not _is_integral(j):
                dim, _ii, _jj, _kk = MID3D, int(i), int(j - 0.5), int(k)
            else:
                dim, _ii, _jj, _kk = INNER, int(i), int(j), int(k - 0.5)
            return arrs[dim][focus[OUTER] + _ii, focus[MID3D] + _jj, focus[INNER] + _kk]

        @staticmethod
        @numba.njit(**jit_flags)
        def atv_axis1(focus, arrs, j, k=0, i=0):
            if not _is_integral(i):
                dim, _i, _jj, _kk = OUTER, int(i - 0.5), int(j), int(k)
            elif not _is_integral(j):
                dim, _i, _jj, _kk = MID3D, int(i), int(j - 0.5), int(k)
            else:
                dim, _i, _jj, _kk = INNER, int(i), int(j), int(k - 0.5)
            return arrs[dim][focus[OUTER] + _i, focus[MID3D] + _jj, focus[INNER] + _kk]

        @staticmethod
        @numba.njit(**jit_flags)
        def atv_axis2(focus, arrs, k, i=0, j=0):
            if not _is_integral(i):
                dim, _i, _j, _k = OUTER, int(i - 0.5), int(j), int(k)
            elif not _is_integral(j):
                dim, _i, _j, _k = MID3D, int(i), int(j - 0.5), int(k)
            else:
                dim, _i, _j, _k = INNER, int(i), int(j), int(k - 0.5)
            return arrs[dim][focus[OUTER] + _i, focus[MID3D] + _j, focus[INNER] + _k]

        @staticmethod
        @numba.njit(**jit_flags)
        def set(arr, i, j, k, value):
            arr[i, j, k] = value

        @staticmethod
        @numba.njit(**jit_flags)
        def get(arr, i, j, k):
            return arr[i, j, k]

    Indexers = namedtuple(
        Path(__file__).stem + "_Indexers", ("ats", "atv", "set", "get")
    )

    indexers = (
        None,
        Indexers((None, None, _1D.ats_1d), (None, None, _1D.atv_1d), _1D.set, _1D.get),
        Indexers(
            (_2D.ats_axis0, None, _2D.ats_axis1),
            (_2D.atv_axis0, None, _2D.atv_axis1),
            _2D.set,
            _2D.get,
        ),
        Indexers(
            (_3D.ats_axis0, _3D.ats_axis1, _3D.ats_axis2),
            (_3D.atv_axis0, _3D.atv_axis1, _3D.atv_axis2),
            _3D.set,
            _3D.get,
        ),
    )
    return indexers

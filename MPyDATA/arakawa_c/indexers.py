"""
Created at 12.03.2020
"""

import numba
from .enumerations import INNER, OUTER, MID3D


@numba.njit([numba.boolean(numba.float64),
             numba.boolean(numba.int64)])
def _is_integral(n):
    return int(n * 2.) % 2 == 0


@numba.njit()
def at_1d(focus, arr, k, _, __):
    return arr[focus[INNER] + k]


@numba.njit()
def at_2d_axis0(focus, arr, i, k, _):
    return arr[focus[OUTER] + i, focus[INNER] + k]


@numba.njit()
def at_2d_axis1(focus, arr, k, i, _):
    return arr[focus[OUTER] + i, focus[INNER] + k]


@numba.njit()
def at_3d_axis0(focus, arr, i, j, k):
    return arr[focus[OUTER] + i, focus[MID3D] + j, focus[INNER] + k]


@numba.njit()
def at_3d_axis1(focus, arr, j, k, i):
    return arr[focus[OUTER] + i, focus[MID3D] + j, focus[INNER] + k]


@numba.njit()
def at_3d_axis1(focus, arr, k, i, j):
    return arr[focus[OUTER] + i, focus[MID3D] + j, focus[INNER] + k]


@numba.njit()
def atv_1d(focus, arrs, k, _, __):
    return arrs[INNER][focus[INNER] + int(k - .5)]


@numba.njit()
def atv_2d_axis0(focus, arrs, i, k, _):
    if _is_integral(i):
        d, ii, kk = INNER, int(i), int(k - .5)
    else:
        d, ii, kk = OUTER, int(i - .5), int(k)
    return arrs[d][focus[OUTER] + ii, focus[INNER] + kk]


@numba.njit()
def atv_2d_axis1(focus, arrs, k, i, _):
    if _is_integral(i):
        d, ii, kk = INNER, int(i), int(k - .5)
    else:
        d, ii, kk = OUTER, int(i - .5), int(k)
    return arrs[d][focus[OUTER] + ii, focus[INNER] + kk]


@numba.njit()
def atv_3d_axis0(focus, arrs, i, j, k):
    if not _is_integral(i):
        d, ii, jj, kk = OUTER, int(i - .5), int(j), int(k)
    elif not _is_integral(j):
        d, ii, jj, kk = MID3D, int(i), int(j - .5), int(k)
    else:
        d, ii, jj, kk = INNER, int(i), int(j), int(k - .5)
    return arrs[d][focus[OUTER] + ii, focus[MID3D] + jj, focus[INNER] + kk]


@numba.njit()
def atv_3d_axis1(focus, arrs, j, k, i):
    if not _is_integral(i):
        d, ii, jj, kk = OUTER, int(i - .5), int(j), int(k)
    elif not _is_integral(j):
        d, ii, jj, kk = MID3D, int(i), int(j - .5), int(k)
    else:
        d, ii, jj, kk = INNER, int(i), int(j), int(k - .5)
    return arrs[d][focus[OUTER] + ii, focus[MID3D] + jj, focus[INNER] + kk]


@numba.njit()
def atv_3d_axis2(focus, arrs, k, i, j):
    if not _is_integral(i):
        d, ii, jj, kk = OUTER, int(i - .5), int(j), int(k)
    elif not _is_integral(j):
        d, ii, jj, kk = MID3D, int(i), int(j - .5), int(k)
    else:
        d, ii, jj, kk = INNER, int(i), int(j), int(k - .5)
    return arrs[d][focus[OUTER] + ii, focus[MID3D] + jj, focus[INNER] + kk]


@numba.njit()
def set_1d(arr, _, __, k, value):
    arr[k] = value


@numba.njit()
def set_2d(arr, i, _, k, value):
    arr[i, k] = value


@numba.njit()
def set_3d(arr, i, j, k, value):
    arr[i, j, k] = value


@numba.njit()
def get_1d(arr, _, __, k):
    return arr[k]


@numba.njit()
def get_2d(arr, i, _, k):
    return arr[i, k]


@numba.njit()
def get_3d(arr, i, j, k):
    return arr[i, j, k]


class Indexers1d:
    at = (None, None, at_1d)
    atv = (None, None, atv_1d)
    set = set_1d
    get = get_1d


class Indexers2d:
    at = (at_2d_axis0, None, at_2d_axis1)
    atv = (atv_2d_axis0, None, atv_2d_axis1)
    set = set_2d
    get = get_2d


class Indexers3d:
    at = (at_3d_axis0, at_3d_axis1, atv_3d_axis2)
    atv = (atv_3d_axis0, atv_3d_axis1, atv_3d_axis2)
    set = set_3d
    get = get_3d


indexers = (None, Indexers1d, Indexers2d, Indexers3d)

"""
Created at 12.03.2020
"""

import numba
import numpy as np
from .enumerations import INNER, OUTER


@numba.njit([numba.boolean(numba.float64),
             numba.boolean(numba.int64)])
def _is_integral(n):
    return int(n * 2.) % 2 == 0


@numba.njit()
def at_1d(focus, arr, j, _):  # TODO: _=0?
    return arr[focus[INNER] + j]


@numba.njit()
def at_2d_axis0(focus, arr, i, j):
    return arr[focus[OUTER] + i, focus[INNER] + j]


@numba.njit()
def at_2d_axis1(focus, arr, i, j):
    return arr[focus[OUTER] + j, focus[INNER] + i]


@numba.njit()
def atv_1d(focus, arrs, j, _):
    return arrs[-1][focus[INNER] + int(j - .5)]


@numba.njit()
def atv_2d_axis0(focus, arrs, i, j):
    if _is_integral(i):
        d = -1
        ii = int(i)
        jj = int(j - .5)
    else:
        d = -2
        ii = int(i - .5)
        jj = int(j)
    return arrs[d][focus[OUTER] + ii, focus[INNER] + jj]


@numba.njit()
def atv_2d_axis1(focus, arrs, i, j):
    if _is_integral(j):
        d = -1
        ii = int(j)
        jj = int(i - .5)
    else:
        d = -2
        ii = int(j - .5)
        jj = int(i)
    return arrs[d][focus[OUTER] + ii, focus[INNER] + jj]


@numba.njit()
def set_1d(arr, _, j, value):
    arr[j] = value


@numba.njit()
def set_2d(arr, i, j, value):
    arr[i, j] = value


@numba.njit()
def get_1d(arr, _, j):
    return arr[j]


@numba.njit()
def get_2d(arr, i, j):
    return arr[i, j]


def make_null():
    return np.empty(0, dtype=np.float64)


class Indexers1d:
    at = (None, at_1d)
    atv = (None, atv_1d)
    set = set_1d
    get = get_1d


class Indexers2d:
    at = (at_2d_axis0, at_2d_axis1)
    atv = (atv_2d_axis0, atv_2d_axis1)
    set = set_2d
    get = get_2d


indexers = (None, Indexers1d, Indexers2d)

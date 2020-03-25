"""
Created at 12.03.2020

@author: Piotr Bartman
"""

import numba
import numpy as np
from MPyDATA.jit_flags import jit_flags

MAX_DIM_NUM = 2


f_i = 0
f_j = f_i + 1
f_d = f_j + 1

@numba.njit([numba.boolean(numba.float64),
             numba.boolean(numba.int64)])
def _is_integral(n):
    return int(n * 2.) % 2 == 0


@numba.njit(**jit_flags)
def at_1d(focus, arr, i, _):
    return arr[focus[f_i] + i]


@numba.njit(**jit_flags)
def at_2d_axis0(focus, arr, i, j):
    return arr[focus[f_i] + i, focus[f_j] + j]


@numba.njit(**jit_flags)
def at_2d_axis1(focus, arr, i, j):
    return arr[focus[f_i] + j, focus[f_j] + i]


@numba.njit(**jit_flags)
def atv_1d(focus, arrs, i, _):
    return arrs[0][focus[f_i] + int(i - .5)]


@numba.njit(**jit_flags)
def atv_2d_axis0(focus, arrs, i, j):
    if _is_integral(i):
        d = 1
        ii = int(i)
        jj = int(j - .5)
    else:
        d = 0
        ii = int(i - .5)
        jj = int(j)
    return arrs[d][focus[f_i] + ii, focus[f_j] + jj]


@numba.njit(**jit_flags)
def atv_2d_axis1(focus, arrs, i, j):
    if _is_integral(j):
        d = 1
        ii = int(j)
        jj = int(i - .5)
    else:
        d = 0
        ii = int(j - .5)
        jj = int(i)
    return arrs[d][focus[f_i] + ii, focus[f_j] + jj]


@numba.njit(**jit_flags)
def set_1d(arr, i, _, value):
    arr[i] = value


@numba.njit(**jit_flags)
def set_2d(arr, i, j, value):
    arr[i, j] = value


@numba.njit(**jit_flags)
def get_1d(arr, i, _):
    return arr[i]


@numba.njit(**jit_flags)
def get_2d(arr, i, j):
    return arr[i, j]


def make_flag(value: bool):
    return np.full(1, value, dtype=bool)


def make_null():
    return np.empty(0, dtype=np.float64)


# TODO: remove at0, at1, ... - use only at[0], at[1]
class Indexers1d:
    at0 = at_1d
    at1 = at_1d  # TODO: redundant
    at = (at_1d, at_1d)
    atv0 = atv_1d
    atv1 = atv_1d  # TODO: redundant
    atv = (atv_1d, atv_1d)
    set = set_1d
    get = get_1d


# TODO: remove at0, at1, ... - use only at[0], at[1]
class Indexers2d:
    at0 = at_2d_axis0
    at1 = at_2d_axis1
    at = (at_2d_axis0, at_2d_axis1)
    atv0 = atv_2d_axis0
    atv1 = atv_2d_axis1
    atv = (atv_2d_axis0, atv_2d_axis1)
    set = set_2d
    get = get_2d


indexers = (None, Indexers1d, Indexers2d)


@numba.njit(**jit_flags)
def null_formula(_, __, ___):
    return 44.

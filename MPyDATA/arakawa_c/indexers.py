"""
Created at 12.03.2020
"""

import numba

MAX_DIM_NUM = 3


f_i = 0
f_j = f_i + 1
f_k = f_j + 1
f_d = f_k + 1


@numba.njit([numba.boolean(numba.float64),
             numba.boolean(numba.int64)])
def _is_integral(n):
    return int(n * 2.) % 2 == 0


@numba.njit()
def at_1d(focus, arr, i, _=0, __=0):
    return arr[focus[f_i] + i]


@numba.njit()
def at_2d_axis0(focus, arr, i, j=0, _=0):
    return arr[focus[f_i] + i, focus[f_j] + j]


@numba.njit()
def at_2d_axis1(focus, arr, i, j=0, _=0):
    return arr[focus[f_i] + j, focus[f_j] + i]


@numba.njit()
def at_3d_axis0(focus, arr, i, j=0, k=0):
    return arr[focus[f_i] + i, focus[f_j] + j, focus[f_k] + k]


@numba.njit()
def at_3d_axis1(focus, arr, i, j=0, k=0):
    return arr[focus[f_i] + j, focus[f_j] + k, focus[f_k] + i]


@numba.njit()
def at_3d_axis2(focus, arr, i, j=0, k=0):
    return arr[focus[f_i] + k, focus[f_j] + i, focus[f_k] + j]


@numba.njit()
def atv_1d(focus, arrs, i, _=0, __=0):
    return arrs[0][focus[f_i] + int(i - .5)]


@numba.njit()
def atv_2d_axis0(focus, arrs, i, j=0, _=0):
    if _is_integral(i):
        d = 1
        ii = int(i)
        jj = int(j - .5)
    else:
        d = 0
        ii = int(i - .5)
        jj = int(j)
    return arrs[d][focus[f_i] + ii, focus[f_j] + jj]


@numba.njit()
def atv_2d_axis1(focus, arrs, i, j=0, _=0):
    if _is_integral(j):
        d = 1
        ii = int(j)
        jj = int(i - .5)
    else:
        d = 0
        ii = int(j - .5)
        jj = int(i)
    return arrs[d][focus[f_i] + ii, focus[f_j] + jj]


@numba.njit()
def atv_3d_axis0(focus, arrs, i, j=0, k=0):
    if not _is_integral(i):
        d = 0
        ii = int(i - .5)
        jj = int(j)
        kk = int(k)
    elif not _is_integral(j):
        d = 1
        ii = int(i)
        jj = int(j - .5)
        kk = int(k)
    else:
        d = 2
        ii = int(i)
        jj = int(j)
        kk = int(k - .5)
    return arrs[d][focus[f_i] + ii, focus[f_j] + jj, focus[f_k] + kk]


@numba.njit()
def atv_3d_axis1(focus, arrs, i, j=0, k=0):
    if not _is_integral(i):
        d = 0
        ii = int(j - .5)
        jj = int(k)
        kk = int(i)
    elif not _is_integral(j):
        d = 1
        ii = int(j)
        jj = int(k - .5)
        kk = int(i)
    else:
        d = 2
        ii = int(j)
        jj = int(k)
        kk = int(i - .5)
    return arrs[d][focus[f_i] + ii, focus[f_j] + jj, focus[f_k] + kk]


@numba.njit()
def atv_3d_axis2(focus, arrs, i, j=0, k=0):
    if not _is_integral(i):
        d = 0
        ii = int(k - .5)
        jj = int(i)
        kk = int(j)
    elif not _is_integral(j):
        d = 1
        ii = int(k)
        jj = int(i - .5)
        kk = int(j)
    else:
        d = 2
        ii = int(k)
        jj = int(i)
        kk = int(j - .5)
    return arrs[d][focus[f_i] + ii, focus[f_j] + jj, focus[f_k] + kk]


@numba.njit()
def set_1d(arr, i, _, __, value):
    arr[i] = value


@numba.njit()
def set_2d(arr, i, j, _, value):
    arr[i, j] = value


@numba.njit()
def set_3d(arr, i, j, k, value):
    arr[i, j, k] = value


@numba.njit()
def get_1d(arr, i, _, __):
    return arr[i]


@numba.njit()
def get_2d(arr, i, j, _):
    return arr[i, j]


@numba.njit()
def get_3d(arr, i, j, k):
    return arr[i, j, k]


class Indexers1d:
    at = (at_1d, None, None)
    atv = (atv_1d, None, None)
    set = set_1d
    get = get_1d


class Indexers2d:
    at = (at_2d_axis0, at_2d_axis1, None)
    atv = (atv_2d_axis0, atv_2d_axis1, None)
    set = set_2d
    get = get_2d


class Indexers3d:
    at = (at_3d_axis0, at_3d_axis1, at_3d_axis2)
    atv = (atv_3d_axis0, atv_3d_axis1, atv_3d_axis2)
    set = set_3d
    get = get_3d


indexers = (None, Indexers1d, Indexers2d, Indexers3d)

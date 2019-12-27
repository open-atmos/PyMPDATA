"""
Created at 07.11.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from MPyDATA.utils import debug_flag

if debug_flag.VALUE:
    import MPyDATA.utils.fake_numba as numba
else:
    import numba


@numba.njit([numba.boolean(numba.float64),
             numba.boolean(numba.int64)])
def _is_integral(n):
    return int(n * 2.) % 2 == 0


@numba.njit([numba.boolean(numba.float64),
             numba.boolean(numba.int64)])
def _is_fractional(n):
    return int(n * 2.) % 2 == 1

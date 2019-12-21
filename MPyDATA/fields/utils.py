"""
Created at 07.11.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from MPyDATA_tests.utils import debug
if debug.DEBUG:
    import MPyDATA_tests.utils.fake_numba as numba
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


def apply(function, output, args, ext=0):
    assert ext < output.halo

    for arg in args:
        arg.fill_halos()

    if len(args) == 1:
        output._apply_1arg(function, args[0], ext)
    elif len(args) == 2:
        output._apply_2arg(function, args[0], args[1], ext)
    elif len(args) == 2:
        output._apply_3arg(function, args[0], args[1], args[2], ext)
    elif len(args) == 4:
        output.apply_4arg(function, args[0], args[1], args[2], args[3], ext)
    else:
        raise NotImplementedError()

    output.invalidate_halos()



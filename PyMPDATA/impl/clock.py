"""CPU-time returning clock() function which works from within njit-ted code,
no time unit guaranteed, returned value only for relative time measurements"""

import ctypes
import sys

import numba
import numpy as np

if sys.version_info < (3, 13):
    clock = ctypes.pythonapi._PyTime_GetSystemClock  # pylint:disable=protected-access
    clock.argtypes = []
    clock.restype = ctypes.c_int
else:
    clock_impl = ctypes.pythonapi.PyTime_Time
    clock_impl.argtypes = [ctypes.c_void_p]
    clock_impl.restype = ctypes.c_int

    assert ctypes.c_time_t == ctypes.c_int64  # pylint: disable=no-member

    @numba.jit("int64()", nopython=True)
    def clock():
        """Numba-JITable version of clock function for Python > 3.12"""
        result = np.empty(shape=(1,), dtype=np.int64)
        clock_impl(result.ctypes)
        return result[0]

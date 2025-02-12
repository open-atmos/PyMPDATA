"""CPU-time returning clock() function which works from within njit-ted code,
no time unit guaranteed, returned value only for relative time measurements"""

import ctypes

clock = ctypes.pythonapi._PyTime_GetSystemClock  # pylint:disable=protected-access
clock.argtypes = []
clock.restype = ctypes.c_int64

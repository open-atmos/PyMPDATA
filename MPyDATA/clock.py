import numba
import ctypes
import platform
from ctypes.util import find_library

if platform.system() == 'Windows':
    libname = 'msvcrt'
else:
    libname = 'c'

clock = ctypes.CDLL(find_library('c')).clock
clock.argtypes = []

scale = 1
if platform.system() == 'Linux':
    scale = 1000


@numba.njit()
def time():
    value = clock()
    return value // scale

import numba
import ctypes
import platform
from ctypes.util import find_library, find_msvcrt

if platform.system() == 'Windows':
    lib = find_msvcrt()
else:
    lib = find_library('c')

clock = ctypes.CDLL(lib).clock
clock.argtypes = []

scale = 1
if platform.system() == 'Linux':
    scale = 1000


@numba.njit()
def time():
    value = clock()
    return value // scale

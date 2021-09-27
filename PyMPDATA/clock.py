import ctypes
import platform

if platform.system() == 'Windows':
    from ctypes.util import find_msvcrt
    __LIB = find_msvcrt()
    if __LIB is None:
        __LIB = 'msvcrt.dll'
else:
    from ctypes.util import find_library
    __LIB = find_library('c')

clock = ctypes.CDLL(__LIB).clock
clock.argtypes = []

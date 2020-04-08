"""
Created at 03.2020
"""

import numba, cffi
import platform

ffi = cffi.FFI()
ffi.cdef('long clock(void);')

sys = platform.system()
scale = 1

if sys == 'Windows':
    libc = ffi.dlopen('msvcrt.dll')
elif sys == 'Linux':
    libc = ffi.dlopen('libc.so.6')
    scale = 1000
elif sys == 'Darwin':
    libc = ffi.dlopen('libc.dylib')
else:
    raise NotImplementedError()

clock = libc.clock

@numba.njit()
def time():
    value = clock()
    return value // scale


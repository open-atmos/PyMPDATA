import numba
import cffi
import platform

ffi = cffi.FFI()
ffi.cdef('long clock(void);')
libc = ffi.dlopen(None)
clock = libc.clock

scale = 1
if platform.system() == 'Linux':
    scale = 1000


@numba.njit()
def time():
    value = clock()
    return value // scale

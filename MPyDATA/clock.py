import numba
import cffi
import platform

sys = platform.system()

ffi = cffi.FFI()
ffi.cdef('long clock(void);')

if sys == 'Windows':
    libc = ffi.dlopen('msvcrt.dll')
else:
    libc = ffi.dlopen(None)

clock = libc.clock

scale = 1
if sys == 'Linux':
    scale = 1000


@numba.njit()
def time():
    value = clock()
    return value // scale

import numba, cffi

ffi = cffi.FFI()
ffi.cdef('long clock(void);')

libc = ffi.dlopen('msvcrt.dll')
clock = libc.clock

@numba.njit()
def time():
    return clock()


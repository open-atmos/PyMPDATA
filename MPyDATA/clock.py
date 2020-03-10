import numba, cffi
import platform

ffi = cffi.FFI()
ffi.cdef('long clock(void);')

sys = platform.system()

if sys == 'Windows':
  libc = ffi.dlopen('msvcrt.dll')
  scale = 1
elif sys == 'Linux' or sys == 'Darwin':
  libc = ffi.dlopen('libc.so.6')
  scale = 1000
else:
  raise NotImplementedError()

clock = libc.clock

@numba.njit()
def time():
    value = clock()
    return value // scale


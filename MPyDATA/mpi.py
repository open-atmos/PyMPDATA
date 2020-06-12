import numba
import cffi
import numpy as np
import platform
from mpi4py import MPI


ffi = cffi.FFI()

if MPI._sizeof(MPI.Comm) == ffi.sizeof('int'):
    _mpi_comm_t = 'int'
else:
    _mpi_comm_t = 'void*'
ffi.cdef(f"typedef {_mpi_comm_t} MPI_Comm;")
ffi.cdef("int MPI_Initialized(int *flag);")
ffi.cdef("int MPI_Comm_size(MPI_Comm comm, int *size);")

lib = None  # libmpi loaded by mpi4py
if platform.system() == 'Windows':
    'libmpi.dll'
libmpi = ffi.dlopen(lib)

_MPI_Comm_size = libmpi.MPI_Comm_size
_MPI_Initialized = libmpi.MPI_Initialized

comm_ptr = MPI._addressof(MPI.COMM_WORLD)
comm_val = ffi.cast('MPI_Comm*', comm_ptr)[0]


@numba.njit()
def initialized():
    flag = np.empty((1,), dtype=np.int32)
    status = _MPI_Initialized(ffi.from_buffer(flag))
    assert status == 0
    return bool(flag[0])


# @numba.njit()
def size():
    value = np.empty(1, dtype=np.int32)
    status = _MPI_Comm_size(comm_val, ffi.from_buffer(value))
    assert status == 0
    return value[0]

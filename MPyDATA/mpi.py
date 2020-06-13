import numba
import ctypes
import numpy as np
import platform
from mpi4py import MPI


if MPI._sizeof(MPI.Comm) == ctypes.sizeof(ctypes.c_int):
    MPI_Comm = ctypes.c_int
else:
    MPI_Comm = ctypes.c_void_p

if platform.system() == 'Linux':
    lib = 'libmpi.so'
elif platform.system() == 'Windows'
    lib = 'libmpi.dll'
elif platform.system() == 'Darwin'
    lib = 'libmpi.dylib'
else:
    raise NotImplementedError()
libmpi = ctypes.CDLL(lib)

_MPI_Initialized = libmpi.MPI_Initialized
_MPI_Initialized.restype = ctypes.c_int
_MPI_Initialized.argtypes = [ctypes.c_void_p]

_MPI_Comm_size = libmpi.MPI_Comm_size
_MPI_Comm_size.restype = ctypes.c_int
_MPI_Comm_size.argtypes = [MPI_Comm, ctypes.c_void_p]

comm_ptr = MPI._addressof(MPI.COMM_WORLD)


# https://stackoverflow.com/questions/61509903/how-to-pass-array-pointer-to-numba-function
@numba.extending.intrinsic
def address_as_void_pointer(typingctx, src):
    """ returns a void pointer from a given memory address """
    from numba.core import types, cgutils
    sig = types.voidptr(src)

    def codegen(cgctx, builder, sig, args):
        return builder.inttoptr(args[0], cgutils.voidptr_t)
    return sig, codegen


@numba.njit()
def initialized():
    flag = np.empty((1,), dtype=np.int32)
    status = _MPI_Initialized(flag.ctypes.data)
    assert status == 0
    return bool(flag[0])


@numba.njit()
def size():
    comm = numba.carray(address_as_void_pointer(comm_ptr), (1,), dtype=np.int64)
    value = np.empty(1, dtype=np.int32)
    status = _MPI_Comm_size(comm[0], value.ctypes.data)
    assert status == 0
    return value[0]

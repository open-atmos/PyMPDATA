import numba
import ctypes
import numpy as np
import platform
from mpi4py import MPI


if MPI._sizeof(MPI.Comm) == ctypes.sizeof(ctypes.c_int):
    _MPI_Comm_t = ctypes.c_int
else:
    _MPI_Comm_t = ctypes.c_void_p

if platform.system() == 'Linux':
    lib = 'libmpi.so'
elif platform.system() == 'Windows':
    lib = 'msmpi.dll'
elif platform.system() == 'Darwin':
    lib = 'libmpi.dylib'
else:
    raise NotImplementedError()
libmpi = ctypes.CDLL(lib)

_MPI_Initialized = libmpi.MPI_Initialized
_MPI_Initialized.restype = ctypes.c_int
_MPI_Initialized.argtypes = [ctypes.c_void_p]

_MPI_Comm_size = libmpi.MPI_Comm_size
_MPI_Comm_size.restype = ctypes.c_int
_MPI_Comm_size.argtypes = [_MPI_Comm_t, ctypes.c_void_p]

_MPI_Comm_rank = libmpi.MPI_Comm_rank
_MPI_Comm_rank.restype = ctypes.c_int
_MPI_Comm_rank.argtypes = [_MPI_Comm_t, ctypes.c_void_p]

_MPI_Comm_World_ptr = MPI._addressof(MPI.COMM_WORLD)


def _MPI_Comm_world():
    return _MPI_Comm_t.from_address(_MPI_Comm_World_ptr)


@numba.extending.overload(_MPI_Comm_world)
def _MPI_Comm_world_njit():
    def impl():
        return numba.carray(
            address_as_void_pointer(_MPI_Comm_World_ptr),
            shape=(1,),
            dtype=np.int64
        )[0]
    return impl


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
    value = np.empty(1, dtype=np.int32)
    status = _MPI_Comm_size(_MPI_Comm_world(), value.ctypes.data)
    assert status == 0
    return value[0]


@numba.njit()
def rank():
    value = np.empty(1, dtype=np.int32)
    status = _MPI_Comm_rank(_MPI_Comm_world(), value.ctypes.data)
    assert status == 0
    return value[0]

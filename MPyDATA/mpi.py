import numba
import ctypes
import numpy as np
import platform
from mpi4py import MPI


if MPI._sizeof(MPI.Comm) == ctypes.sizeof(ctypes.c_int):
    _MPI_Comm_t = ctypes.c_int
else:
    _MPI_Comm_t = ctypes.c_void_p
_MPI_Datatype_t = ctypes.c_int
_MPI_Status_ptr_t = ctypes.c_void_p

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

# int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,  MPI_Comm comm)
_MPI_Send = libmpi.MPI_Send
_MPI_Send.restype = ctypes.c_int
_MPI_Send.argtypes = [ctypes.c_void_p, ctypes.c_int, _MPI_Datatype_t, ctypes.c_int, ctypes.c_int, _MPI_Comm_t]

#int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status * status)
_MPI_Recv = libmpi.MPI_Recv
_MPI_Recv.restype = ctypes.c_int
_MPI_Recv.argtypes = [ctypes.c_void_p, ctypes.c_int, _MPI_Datatype_t, ctypes.c_int, ctypes.c_int, _MPI_Comm_t, _MPI_Status_ptr_t]

def _MPI_Comm_world():
    return _MPI_Comm_t.from_address(_MPI_Comm_World_ptr)


@numba.extending.overload(_MPI_Comm_world)
def _MPI_Comm_world_njit():
    def impl():
        return numba.carray(
            address_as_void_pointer(_MPI_Comm_World_ptr),
            shape=(1,),
            dtype=np.intp
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
    flag = np.empty((1,), dtype=np.intc)
    status = _MPI_Initialized(flag.ctypes.data)
    assert status == 0
    return bool(flag[0])


@numba.njit()
def size():
    value = np.empty(1, dtype=np.intc)
    status = _MPI_Comm_size(_MPI_Comm_world(), value.ctypes.data)
    assert status == 0
    return value[0]


@numba.njit()
def rank():
    value = np.empty(1, dtype=np.intc)
    status = _MPI_Comm_rank(_MPI_Comm_world(), value.ctypes.data)
    assert status == 0
    return value[0]

# TEMP: from WIN MSDN (https://docs.microsoft.com/en-us/message-passing-interface/mpi-datatype-enumeration)
MPI_DATATYPE_NULL          = 0x0c000000
MPI_CHAR                   = 0x4c000101
MPI_UNSIGNED_CHAR          = 0x4c000102
MPI_SHORT                  = 0x4c000203
MPI_UNSIGNED_SHORT         = 0x4c000204
MPI_INT                    = 0x4c000405
MPI_UNSIGNED               = 0x4c000406
MPI_LONG                   = 0x4c000407
MPI_UNSIGNED_LONG          = 0x4c000408
MPI_LONG_LONG_INT          = 0x4c000809
MPI_LONG_LONG              = MPI_LONG_LONG_INT
MPI_FLOAT                  = 0x4c00040a
MPI_DOUBLE                 = 0x4c00080b
MPI_LONG_DOUBLE            = 0x4c00080c
MPI_BYTE                   = 0x4c00010d
MPI_WCHAR                  = 0x4c00020e
MPI_PACKED                 = 0x4c00010f
MPI_LB                     = 0x4c000010
MPI_UB                     = 0x4c000011
MPI_C_COMPLEX              = 0x4c000812
MPI_C_FLOAT_COMPLEX        = 0x4c000813
MPI_C_DOUBLE_COMPLEX       = 0x4c001614
MPI_C_LONG_DOUBLE_COMPLEX  = 0x4c001615
MPI_2INT                   = 0x4c000816
MPI_C_BOOL                 = 0x4c000117
MPI_SIGNED_CHAR            = 0x4c000118
MPI_UNSIGNED_LONG_LONG     = 0x4c000819
MPI_CHARACTER              = 0x4c00011a
MPI_INTEGER                = 0x4c00041b
MPI_REAL                   = 0x4c00041c
MPI_LOGICAL                = 0x4c00041d
MPI_COMPLEX                = 0x4c00081e
MPI_DOUBLE_PRECISION       = 0x4c00081f
MPI_2INTEGER               = 0x4c000820
MPI_2REAL                  = 0x4c000821
MPI_DOUBLE_COMPLEX         = 0x4c001022
MPI_2DOUBLE_PRECISION      = 0x4c001023
MPI_2COMPLEX               = 0x4c001024
MPI_2DOUBLE_COMPLEX        = 0x4c002025
MPI_REAL2                  = MPI_DATATYPE_NULL
MPI_REAL4                  = 0x4c000427
MPI_COMPLEX8               = 0x4c000828
MPI_REAL8                  = 0x4c000829
MPI_COMPLEX16              = 0x4c00102a
MPI_REAL16                 = MPI_DATATYPE_NULL
MPI_COMPLEX32              = MPI_DATATYPE_NULL
MPI_INTEGER1               = 0x4c00012d
MPI_COMPLEX4               = MPI_DATATYPE_NULL
MPI_INTEGER2               = 0x4c00022f
MPI_INTEGER4               = 0x4c000430
MPI_INTEGER8               = 0x4c000831
MPI_INTEGER16              = MPI_DATATYPE_NULL
MPI_INT8_T                 = 0x4c000133
MPI_INT16_T                = 0x4c000234
MPI_INT32_T                = 0x4c000435
MPI_INT64_T                = 0x4c000836
MPI_UINT8_T                = 0x4c000137
MPI_UINT16_T               = 0x4c000238
MPI_UINT32_T               = 0x4c000439
MPI_UINT64_T               = 0x4c00083a
MPI_AINT                   = 0x4c00083b
MPI_OFFSET                 = 0x4c00083c
MPI_FLOAT_INT              = 0x8c000000
MPI_DOUBLE_INT             = 0x8c000001
MPI_LONG_INT               = 0x8c000002
MPI_SHORT_INT              = 0x8c000003
MPI_LONG_DOUBLE_INT        = 0x8c000004

#TEMP: Datatype mapping
_TYPES_NP2MPI_RAW = {
    np.int8: MPI_INT8_T,
    np.int16: MPI_INT16_T,
    np.int32: MPI_INT32_T,
    np.int64: MPI_INT64_T,
    np.uint8: MPI_UINT8_T,
    np.uint16: MPI_UINT16_T,
    np.uint32: MPI_UINT32_T,
    np.uint64: MPI_UINT64_T,
    np.float32: MPI_FLOAT,
    np.float64: MPI_DOUBLE,
    np.float_: MPI_DOUBLE,
    np.complex64: MPI_COMPLEX,
    np.complex128: MPI_DOUBLE_COMPLEX,
    np.complex_: MPI_C_LONG_DOUBLE_COMPLEX,
    np.bool_: MPI_CHAR,
    np.byte: MPI_CHAR,
    np.ubyte: MPI_UNSIGNED_CHAR,
    np.short: MPI_SHORT,
    np.ushort: MPI_UNSIGNED_SHORT,
    np.intc: MPI_INT,
    np.uintc: MPI_UNSIGNED,
    np.int_: MPI_LONG,
    np.uint: MPI_UNSIGNED_LONG,
    np.longlong: MPI_LONG_LONG,
    np.ulonglong: MPI_UNSIGNED_LONG_LONG,
    np.half: MPI_REAL2,
    np.float16: MPI_REAL2,
    np.intp: MPI_DATATYPE_NULL, # ???
    np.uintp: MPI_DATATYPE_NULL, # ???
    np.single: MPI_FLOAT,
    np.double: MPI_DOUBLE,
    np.longdouble: MPI_LONG_DOUBLE,
    np.csingle: MPI_FLOAT,
    np.cdouble: MPI_DOUBLE,
    np.clongdouble: MPI_LONG_DOUBLE}

_NP2MPI_max_num = max(np.dtype(dt).num for dt in _TYPES_NP2MPI_RAW)
_TYPES_NP2MPI = np.empty(_NP2MPI_max_num + 1, dtype=np.intc)
for k, v in _TYPES_NP2MPI_RAW.items():
    _TYPES_NP2MPI[np.dtype(k).num] = v

@numba.njit
def _MPI_get_type_as_int(dtype):
    return _TYPES_NP2MPI[dtype.num].ctypes.data

# int MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag,  MPI_Comm comm)
@numba.njit
def send(data, dest, tag):
    result = _MPI_Send(x.ctypes.data, x.size, _MPI_get_type_as_int(x.dtype), dest, tag, _MPI_Comm_world())
    assert result == 0

#int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status * status)
@numba.njit()
def recv(data, source, tag):
    status = np.empty((5,), dtype=np.intc)
    result = _MPI_Recv(x.ctypes.data, x.size, _MPI_get_type_as_int(x.dtype), source, tag, _MPI_Comm_world(), status.ctypes.data)

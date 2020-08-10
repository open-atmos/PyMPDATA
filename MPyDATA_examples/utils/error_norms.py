import numpy as np
import numba


@numba.njit()
def L2(numerical, analytical, nt):
    assert numerical.shape == analytical.shape
    N = analytical.size
    err2 = np.log2(
        np.sqrt(
            np.sum((numerical - analytical)**2) / nt / N
        )
    )
    return err2


@numba.njit()
def GMD(numerical, analytical, T):
    assert numerical.shape == analytical.shape
    NX = analytical.size
    err = np.sqrt(
            np.sum((numerical - analytical)**2) / NX
        ) / T
    return err
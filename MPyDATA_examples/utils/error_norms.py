import numpy as np


def L2(numerical, analytical, nt):
    assert numerical.shape == analytical.shape
    N = analytical.size
    err2 = np.log(
        np.sqrt(
            sum(pow(numerical - analytical, 2)) / nt / N
        )
    ) / np.log(2)
    return err2

def GMD(numerical, analytical, T):
    assert numerical.shape == analytical.shape
    NX = analytical.size
    err = np.sqrt(
            sum(pow(numerical - analytical, 2)) / NX
        ) / T
    return err
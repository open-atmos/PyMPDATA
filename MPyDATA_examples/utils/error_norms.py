import numpy as np


def L2(numerical, analytical, nt, nx):
    assert numerical.shape == analytical.shape  # TODO: == nx
    err2 = np.log(
        np.sqrt(
            sum(pow(numerical - analytical, 2)) / nt / nx
        )
    ) / np.log(2)
    return err2

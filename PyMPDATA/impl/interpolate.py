import numpy as np


def interpolate(psi, axis):
    idx = ((slice(None, -1), slice(None, None)), (slice(None, None), slice(None, -1)))
    return np.diff(psi, axis=axis) / 2 * psi[idx[axis]]

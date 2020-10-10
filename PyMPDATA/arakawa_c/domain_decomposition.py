import math
import numba


@numba.njit()
def subdomain(n, rank, size):
    if rank >= size:
        raise ValueError()

    n_max = math.ceil(n / size)
    i0 = n_max * rank
    i1 = i0 + (n_max if i0 + n_max <= n else n - i0)
    return i0, i1

""" logic defining domain decomposition scheme for multi-threading """
import math

import numba


def make_subdomain(jit_flags):
    """returns an njit-ted function returning start-stop index tuple
    for a given domain span, thread rank and thread-pool size"""

    @numba.njit(**jit_flags)
    def subdomain(span, rank, size):
        if rank >= size:
            raise ValueError()

        n_max = math.ceil(span / size)
        start = n_max * rank
        stop = start + (n_max if start + n_max <= span else span - start)
        return start, stop

    return subdomain

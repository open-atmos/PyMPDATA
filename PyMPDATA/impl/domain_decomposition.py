import math
import numba

def make_subdomain(jit_flags):
    @numba.njit(**jit_flags)
    def subdomain(n, rank, size):
        if rank >= size:
            raise ValueError()

        n_max = math.ceil(n / size)
        i0 = n_max * rank
        i1 = i0 + (n_max if i0 + n_max <= n else n - i0)
        return i0, i1
    return subdomain

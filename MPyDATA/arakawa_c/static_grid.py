import numba

from MPyDATA.arakawa_c.domain_decomposition import subdomain
from MPyDATA.arakawa_c.meta import meta_nouter, meta_ninner


def make_chunk(n, n_threads):
    static = n > 0

    if static:
        rngs = tuple([subdomain(n, th, n_threads) for th in range(n_threads)])

        @numba.njit()
        def _impl(_, thread_id):
            return rngs[thread_id]
    else:
        @numba.njit()
        def _impl(meta, thread_id):
            return subdomain(meta[meta_nouter], thread_id, n_threads)

    return _impl


def make_domain(grid):
    static = grid[0] > 0

    if static:
        @numba.njit()
        def _impl(_):
            return grid
    else:
        @numba.njit()
        def _impl(meta):
            return meta[meta_nouter], meta[meta_ninner]
    return _impl
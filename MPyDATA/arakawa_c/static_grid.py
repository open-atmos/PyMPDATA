import numba

from MPyDATA.arakawa_c.domain_decomposition import subdomain
from MPyDATA.arakawa_c.meta import meta_ni, meta_nj


def make_irng(ni, n_threads):
    static = ni > 0

    if static:
        rngs = tuple([subdomain(ni, th, n_threads) for th in range(n_threads)])

        @numba.njit()
        def _impl(_, thread_id):
            return rngs[thread_id]
    else:
        @numba.njit()
        def _impl(meta, thread_id):
            return subdomain(meta[meta_ni], thread_id, n_threads)

    return _impl


def make_grid(grid):
    static = grid[0] > 0

    if static:
        @numba.njit()
        def _impl(_):
            return grid
    else:
        @numba.njit()
        def _impl(meta):
            return meta[meta_ni], meta[meta_nj]
    return _impl
import numba

from ..arakawa_c.domain_decomposition import subdomain
from ..arakawa_c.meta import META_N_OUTER, META_N_MID3D, META_N_INNER


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
            return subdomain(meta[META_N_OUTER], thread_id, n_threads)

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
            return meta[META_N_OUTER], meta[META_N_MID3D], meta[META_N_INNER]
    return _impl

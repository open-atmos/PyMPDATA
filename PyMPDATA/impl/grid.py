"""
static (extents known to JIT) and dynamic (run-time extents) grid handling logic
"""
import numba

from PyMPDATA.impl.domain_decomposition import make_subdomain
from PyMPDATA.impl.meta import META_N_INNER, META_N_MID3D, META_N_OUTER


def make_chunk(span, n_threads, jit_flags):
    """returns an njit-ted function which returns the subdomain extent for a
    given thread, for static grid size no calculations are done at run-time"""
    static = span > 0

    subdomain = make_subdomain(jit_flags)

    if static:
        rngs = tuple(subdomain(span, th, n_threads) for th in range(n_threads))

        @numba.njit(**jit_flags)
        def _impl(_, thread_id):
            return rngs[thread_id]

    else:

        @numba.njit(**jit_flags)
        def _impl(meta, thread_id):
            return subdomain(meta[META_N_OUTER], thread_id, n_threads)

    return _impl


def make_domain(grid, jit_flags):
    """returns an njit-ted function which, for static grids, returns a compile-time-constant
    grid size, and otherwise returns the grid size encoded in the meta tuple"""
    static = grid[0] > 0

    if static:

        @numba.njit(**jit_flags)
        def _impl(_):
            return grid

    else:

        @numba.njit(**jit_flags)
        def _impl(meta):
            return meta[META_N_OUTER], meta[META_N_MID3D], meta[META_N_INNER]

    return _impl

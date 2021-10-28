""" commons for scalar and vector field traversals """
import numba
from .enumerations import RNG_STOP, OUTER


def _make_common(jit_flags, spanner, chunker):
    @numba.njit(**jit_flags)
    def common(meta, thread_id):
        span = spanner(meta)
        rng_outer = chunker(meta, thread_id)
        last_thread = rng_outer[RNG_STOP] == span[OUTER]
        first_thread = thread_id == 0
        return span, rng_outer, last_thread, first_thread
    return common

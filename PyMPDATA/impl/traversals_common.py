""" commons for scalar and vector field traversals """
# pylint: disable=too-many-arguments
import numba

from .enumerations import OUTER, RNG_STOP


def make_common(jit_flags, spanner, chunker):
    """returns Numba-compiled callable producing common parameters"""
    @numba.njit(**jit_flags)
    def common(meta, thread_id):
        span = spanner(meta)
        rng_outer = chunker(meta, thread_id)
        last_thread = rng_outer[RNG_STOP] == span[OUTER]
        first_thread = thread_id == 0
        return span, rng_outer, last_thread, first_thread

    return common


def make_fill_halos_loop(jit_flags, set_value, fill_halos):
    """returns Numba-compiled halo-filling callable"""
    @numba.njit(**jit_flags)
    def fill_halos_loop(i_rng, j_rng, k_rng, psi, span, sign):
        for i in i_rng:
            for j in j_rng:
                for k in k_rng:
                    focus = (i, j, k)
                    set_value(psi, *focus, fill_halos((focus, psi), span, sign))

    return fill_halos_loop

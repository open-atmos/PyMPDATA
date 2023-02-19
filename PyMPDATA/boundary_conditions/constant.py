""" boundary condition filling halos with a constant value """
from functools import lru_cache

import numba


class Constant:
    """class which instances are to be passed in boundary_conditions tuple to the
    `PyMPDATA.scalar_field.ScalarField` and
    `PyMPDATA.vector_field.VectorField` __init__ methods"""

    def __init__(self, value):
        self.value = value

    def make_scalar(self, ats, set_value, halo, dtype, jit_flags):
        """returns (lru-cached) Numba-compiled scalar halo-filling callable"""
        return _make_scalar_constant(self.value, ats, set_value, halo, dtype, jit_flags)

    def make_vector(self, ats, set_value, halo, dtype, jit_flags):
        """returns (lru-cached) Numba-compiled vector halo-filling callable"""
        return _make_scalar_constant(self.value, ats, set_value, halo, dtype, jit_flags)


@lru_cache()
def _make_scalar_constant(value, _, set_value, __, ___, jit_flags):
    @numba.njit(**jit_flags)
    def fill_halos(_1, _2, _3):
        return value

    # <TODO>
    @numba.njit(**jit_flags)
    def fill_halos_loop(i_rng, j_rng, k_rng, psi, span, sign):
        for i in i_rng:
            for j in j_rng:
                for k in k_rng:
                    focus = (i, j, k)
                    set_value(psi, *focus, fill_halos((focus, psi), span, sign))

    # </TODO>

    return fill_halos_loop

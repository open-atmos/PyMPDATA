import numba


@numba.njit()
def null_scalar_formula(_, __, ___):
    return 44.


@numba.njit()
def null_vector_formula(_, __, ___, ____):
    return 666.

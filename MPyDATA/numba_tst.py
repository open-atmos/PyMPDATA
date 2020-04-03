import numba

MAX_DIM_NUM = 2


@numba.njit()
def apply_scalar(fun_0, fun_1):
    return 0


def make_upwind():

    formulae_upwind = (__make_upwind(), __make_upwind())

    @numba.njit()
    def apply():
        return apply_scalar(*formulae_upwind)

    return apply


def __make_upwind():
    @numba.njit()
    def upwind():
        return 0
    return upwind

upwind = make_upwind()
upwind()

import numba


@numba.njit()
def apply(fun_0, fun_1):
    return 0


def make_bar():

    formulae_foo = (__make_foo(), __make_foo())

    @numba.njit()
    def bar():
        return apply(*formulae_foo)

    return bar


def __make_foo():
    @numba.njit()
    def foo():
        return 0
    return foo


fun = make_bar()
fun()

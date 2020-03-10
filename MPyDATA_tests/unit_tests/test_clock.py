import numba
from MPyDATA.clock import clock


def test_clock_python():
    clock()


def test_clock_numba_jit():
    @numba.jit()
    def test():
        clock()
    test()


def test_clock_numba_njit():
    @numba.njit()
    def test():
        clock()
    test()


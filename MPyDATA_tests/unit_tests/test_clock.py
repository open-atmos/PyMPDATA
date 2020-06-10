import numba
from MPyDATA.clock import clock


class TestClock:
    @staticmethod
    def test_clock_python():
        clock()

    @staticmethod
    def test_clock_numba_jit():
        @numba.jit()
        def test():
            clock()
        test()

    @staticmethod
    def test_clock_numba_njit():
        @numba.njit()
        def test():
            clock()
        test()

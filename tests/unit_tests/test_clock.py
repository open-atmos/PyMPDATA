# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import time
import numba
from PyMPDATA.options import Options
from PyMPDATA.impl.clock import clock

jit_flags = Options().jit_flags


class TestClock:
    @staticmethod
    def test_clock_python():
        clock()

    @staticmethod
    def test_clock_numba_jit():
        @numba.jit(**jit_flags)
        def test():
            clock()
        test()

    @staticmethod
    def test_clock_numba_njit():
        @numba.njit(**jit_flags)
        def test():
            clock()
        test()

    @staticmethod
    def test_clock_value():
        # Arrange
        sec_expected = 2
        t0 = clock()

        # Act
        time.sleep(sec_expected)
        t1 = clock()

        # Assert
        sec_actual = (t1 - t0) / 1000
        assert (sec_actual - sec_expected) / sec_expected < 0.1

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import time

import numba

from PyMPDATA.impl.clock import clock
from PyMPDATA.options import Options

jit_flags = Options().jit_flags


class TestClock:
    @staticmethod
    def test_clock_python():
        clock()

    @staticmethod
    def test_clock_numba_jit():
        @numba.jit(**{**jit_flags, "forceobj": True})
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
        factor = 3
        base = 0.1

        sec_base = None
        warmup = 1
        for _ in range(warmup + 1):
            start = clock()
            time.sleep(base)
            sec_base = clock() - start

        # Act
        start = clock()
        time.sleep(base * factor)
        sec_factor = clock() - start

        # Assert
        assert abs(sec_factor / sec_base - factor) < 0.1

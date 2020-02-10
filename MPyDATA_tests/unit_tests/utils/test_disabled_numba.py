from MPyDATA.utils.disabled_numba import DisabledNumba
from MPyDATA.formulae.antidiff import make_antidiff
from MPyDATA.arakawa_c.impl import scalar_field_2d
from MPyDATA.options import Options
from MPyDATA.mpdata_factory import MPDATAFactory
import numpy as np
import pytest


class TestDisabledNumba:
    @staticmethod
    def test_function():
        # Arrange
        sut = DisabledNumba()
        fun = lambda: make_antidiff(Options())

        # Act & Assert
        assert hasattr(fun(), "py_func")
        with sut:
            from MPyDATA.utils.debug_flag import VALUE
            assert VALUE
            assert not hasattr(fun(), "py_func")
        assert hasattr(fun(), "py_func")

    @staticmethod
    def test_class():
        # Arrange
        sut = DisabledNumba()
        cls = lambda: scalar_field_2d.make_scalar_field_2d(np.eye(3), 2)

        # Act & Assert
        assert "numba" in str(cls())
        with sut:
            assert not "numba" in str(cls())
            from MPyDATA.utils.debug_flag import VALUE
            assert VALUE
        assert "numba" in str(cls())

    @pytest.mark.skip # TODO: should step() be jitted?
    @staticmethod
    def test_method():
        sut = MPDATAFactory.uniform_C_1d(np.array([0, 1, 0]), 0, Options(), None)
        assert hasattr(sut.step, "py_func")

        with DisabledNumba():
            sut = MPDATAFactory.uniform_C_1d(np.array([0, 1, 0]), 0, Options(), None)
            assert not hasattr(sut.step, "py_func")

        sut = MPDATAFactory.uniform_C_1d(np.array([0, 1, 0]), 0, Options(), None)
        assert hasattr(sut.step, "py_func")



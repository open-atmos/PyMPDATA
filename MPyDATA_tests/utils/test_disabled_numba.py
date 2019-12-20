from MPyDATA_tests.utils.disabled_numba import DisabledNumba
from MPyDATA.formulae.antidiff import make_antidiff
from MPyDATA.fields import _scalar_field_2d
from MPyDATA.options import Options


class TestDisabledNumba:
    @staticmethod
    def test_function():
        # Arrange
        sut = DisabledNumba()
        fun = lambda: make_antidiff(Options())

        # Act & Assert
        assert hasattr(fun(), "py_func")
        with sut:
            from MPyDATA_tests.utils.debug import DEBUG
            assert DEBUG
            assert not hasattr(fun(), "py_func")
        assert hasattr(fun(), "py_func")

    @staticmethod
    def test_class():
        # Arrange
        sut = DisabledNumba()
        cls = lambda: _scalar_field_2d.ScalarField2D

        # Act & Assert
        assert "numba" in str(cls())
        with sut:
            assert not "numba" in str(cls())
            from MPyDATA_tests.utils.debug import DEBUG
            assert DEBUG
        assert "numba" in str(cls())

    @staticmethod
    def test_TODO():
        from MPyDATA.mpdata_factory import MPDATAFactory
        import numpy as np

        with DisabledNumba():
            mpdata = MPDATAFactory.uniform_C_1d(np.array([0, 1, 0]), 0, Options())
            assert not hasattr(mpdata.step, "py_func")


from MPyDATA.utils.disabled_numba import DisabledNumba
from MPyDATA.formulae.antidiff import make_antidiff
from MPyDATA.opts import Opts


def test_fake_numba():
    # Arrange
    sut = DisabledNumba()
    fun = lambda: make_antidiff(Opts())

    # Act & Assert
    assert hasattr(fun(), "py_func")
    with sut:
        from MPyDATA.utils.debug import DEBUG
        assert DEBUG == True
        assert not hasattr(fun(), "py_func")
    assert hasattr(fun(), "py_func")

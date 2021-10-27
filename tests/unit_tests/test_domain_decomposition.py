# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import pytest
from PyMPDATA import Options
from PyMPDATA.impl.domain_decomposition import make_subdomain


JIT_FLAGS = Options().jit_flags


@pytest.mark.parametrize("span, rank, size, result", [
    (10, 0, 1, (0, 10)),
    pytest.param(1, 1, 1, (0, 1), marks=pytest.mark.xfail(raises=ValueError)),
    (10, 0, 3, (0, 4)),
    (10, 1, 3, (4, 8)),
    (10, 2, 3, (8, 10)),
    (10, 0, 11, (0, 1)),
    (10, 9, 11, (9, 10))
])
def test_subdomain(span, rank, size, result):
    subdomain = make_subdomain(JIT_FLAGS)
    assert subdomain(span, rank, size) == result

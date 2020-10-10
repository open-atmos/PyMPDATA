from PyMPDATA.arakawa_c.domain_decomposition import subdomain
import pytest


@pytest.mark.parametrize("n, rank, size, range", [
    (10, 0, 1, (0, 10)),
    pytest.param(1, 1, 1, (0, 1), marks=pytest.mark.xfail(raises=ValueError)),
    (10, 0, 3, (0, 4)),
    (10, 1, 3, (4, 8)),
    (10, 2, 3, (8, 10)),
    (10, 0, 11, (0, 1)),
    (10, 9, 11, (9, 10))
])
def test_subdomain(n, rank, size, range):
    assert subdomain(n, rank, size) == range

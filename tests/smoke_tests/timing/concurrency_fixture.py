import numba
import pytest


__concurrency_str = ("threads", "serial")
try:
    numba.parfors.parfor.ensure_parallel_support()
except numba.core.errors.UnsupportedParforsError:
    __concurrency_str = ("serial",)


@pytest.fixture(params=__concurrency_str)
def concurrency(request):
    return request.param

# pylint: disable=missing-module-docstring,missing-function-docstring
import numba
import pytest

__n_threads = (1, 2, 3)
try:
    numba.parfors.parfor.ensure_parallel_support()
except numba.core.errors.UnsupportedParforsError:
    __n_threads = (1,)


@pytest.fixture(params=__n_threads, name="n_threads")
def n_threads_fixture(request):
    return request.param

import numba
import pytest

__num_threads = [pytest.param(1, id='serial')]

try:
    numba.parfors.parfor.ensure_parallel_support()
    n = numba.config.NUMBA_NUM_THREADS
    __num_threads.append(pytest.param(n, id=f"threads ({n})"))
except numba.core.errors.UnsupportedParforsError:
    pass


@pytest.fixture(params=__num_threads)
def num_threads(request):
    return request.param

""" pytest fixtures for performance tests """
import numba
import pytest

__num_threads = [pytest.param(1, id="serial")]

try:
    numba.parfors.parfor.ensure_parallel_support()
    n = numba.config.NUMBA_NUM_THREADS  # pylint: disable=no-member
    assert n != 1
    __num_threads.append(pytest.param(n, id=f"threads ({n})"))
except numba.core.errors.UnsupportedParforsError:
    pass


@pytest.fixture(params=__num_threads)
def num_threads(request):
    """pytest fixture providing thread-pool size for tests: single-thread case
    for setups in which Numba reports no parallel support, and single-
    as well as multi-threaded test runs otherwise. For the multi-threaded
    case, the number of threads is set to NUMBA_NUM_THREADS."""
    return request.param

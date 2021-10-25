# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import pytest
from PyMPDATA import Options
from PyMPDATA.impl.grid import make_domain, make_chunk
from PyMPDATA.impl.meta import META_N_OUTER, META_N_MID3D, META_N_INNER, META_SIZE
from PyMPDATA.impl.domain_decomposition import make_subdomain

meta = [None] * META_SIZE
meta[META_N_OUTER] = 200
meta[META_N_MID3D] = 0
meta[META_N_INNER] = 2000
meta = tuple(meta)

JIT_FLAGS = Options().jit_flags


class TestStaticGrid:
    @staticmethod
    def test_make_grid_static():
        # arrange
        grid = (100, 1000)
        assert grid[0] != meta[META_N_OUTER]
        assert grid[0] != meta[META_N_INNER]

        # act
        grid_fun = make_domain(grid, jit_flags=JIT_FLAGS)

        # assert
        assert grid == grid_fun(meta)

    @staticmethod
    def test_make_grid_dynamic():
        # arrange
        grid = (0, )

        # act
        grid_fun = make_domain(grid, jit_flags=JIT_FLAGS)

        # assert
        assert (meta[META_N_OUTER], meta[META_N_MID3D], meta[META_N_INNER]) == grid_fun(meta)

    @staticmethod
    @pytest.mark.parametrize("span", (3, 30, 300))
    @pytest.mark.parametrize("n_threads", (1, 2, 3))
    def test_make_irng_static(span, n_threads):
        # arrange
        assert span != meta[META_N_OUTER]
        subdomain = make_subdomain(JIT_FLAGS)

        # act
        irng_fun = make_chunk(span=span, n_threads=n_threads, jit_flags=JIT_FLAGS)

        # assert
        for thread_id in range(n_threads):
            assert subdomain(span, thread_id, n_threads) == irng_fun(meta, thread_id)

    @staticmethod
    @pytest.mark.parametrize("n_threads", (1, 2, 3))
    def test_make_irng_dynamic(n_threads):
        # arrange
        span = 0
        subdomain = make_subdomain(JIT_FLAGS)

        # act
        irng_fun = make_chunk(span=span, n_threads=n_threads, jit_flags=JIT_FLAGS)

        # assert
        for thread_id in range(n_threads):
            assert subdomain(meta[META_N_OUTER], thread_id, n_threads) == irng_fun(meta, thread_id)

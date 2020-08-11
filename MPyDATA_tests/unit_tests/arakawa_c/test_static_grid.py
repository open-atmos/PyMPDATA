from MPyDATA.arakawa_c.static_grid import make_domain, make_chunk
from MPyDATA.arakawa_c.meta import meta_ni, meta_nj, meta_nk, meta_size
from MPyDATA.arakawa_c.domain_decomposition import subdomain
import pytest

meta = [None] * meta_size
meta[meta_ni] = 200
meta[meta_nj] = 2000
meta[meta_nk] = 0
meta = tuple(meta)


class TestStaticGrid:
    @staticmethod
    def test_make_grid_static():
        # arrange
        grid = (100, 1000)
        assert grid[0] != meta[meta_ni]

        # act
        grid_fun = make_domain(grid)

        # assert
        assert grid == grid_fun(meta)

    @staticmethod
    def test_make_grid_dynamic():
        # arrange
        grid = (0, )

        # act
        grid_fun = make_domain(grid)

        # assert
        assert (meta[meta_ni], meta[meta_nj], meta[meta_nk]) == grid_fun(meta)

    @staticmethod
    @pytest.mark.parametrize("ni", (3, 30, 300))
    @pytest.mark.parametrize("n_threads", (1, 2, 3))
    def test_make_irng_static(ni, n_threads):
        # arrange
        assert ni != meta[meta_ni]

        # act
        irng_fun = make_chunk(ni=ni, n_threads=n_threads)

        # assert
        for thread_id in range(n_threads):
            assert subdomain(ni, thread_id, n_threads) == irng_fun(meta, thread_id)

    @staticmethod
    @pytest.mark.parametrize("n_threads", (1, 2, 3))
    def test_make_irng_dynamic(n_threads):
        # arrange
        ni = 0

        # act
        irng_fun = make_chunk(ni=ni, n_threads=n_threads)

        # assert
        for thread_id in range(n_threads):
            assert subdomain(meta[meta_ni], thread_id, n_threads) == irng_fun(meta, thread_id)


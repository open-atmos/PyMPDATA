from MPyDATA.arakawa_c.static_grid import make_domain, make_chunk
from MPyDATA.arakawa_c.meta import meta_nouter, meta_ninner, meta_size
from MPyDATA.arakawa_c.domain_decomposition import subdomain
import pytest

meta = [None] * meta_size
meta[meta_nouter] = 200
meta[meta_ninner] = 2000
meta = tuple(meta)


class TestStaticGrid:
    @staticmethod
    def test_make_grid_static():
        # arrange
        grid = (100, 1000)
        assert grid[0] != meta[meta_nouter]
        assert grid[0] != meta[meta_ninner]

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
        assert (meta[meta_nouter], meta[meta_ninner]) == grid_fun(meta)

    @staticmethod
    @pytest.mark.parametrize("n", (3, 30, 300))
    @pytest.mark.parametrize("n_threads", (1, 2, 3))
    def test_make_irng_static(n, n_threads):
        # arrange
        assert n != meta[meta_nouter]

        # act
        irng_fun = make_chunk(n=n, n_threads=n_threads)

        # assert
        for thread_id in range(n_threads):
            assert subdomain(n, thread_id, n_threads) == irng_fun(meta, thread_id)

    @staticmethod
    @pytest.mark.parametrize("n_threads", (1, 2, 3))
    def test_make_irng_dynamic(n_threads):
        # arrange
        n = 0

        # act
        irng_fun = make_chunk(n=n, n_threads=n_threads)

        # assert
        for thread_id in range(n_threads):
            assert subdomain(meta[meta_nouter], thread_id, n_threads) == irng_fun(meta, thread_id)


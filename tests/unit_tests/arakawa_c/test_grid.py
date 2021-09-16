from PyMPDATA.arakawa_c.grid import make_domain, make_chunk
from PyMPDATA.arakawa_c.meta import META_N_OUTER, META_N_MID3D, META_N_INNER, META_SIZE
from PyMPDATA.arakawa_c.domain_decomposition import subdomain
import pytest

meta = [None] * META_SIZE
meta[META_N_OUTER] = 200
meta[META_N_MID3D] = 0
meta[META_N_INNER] = 2000
meta = tuple(meta)


class TestStaticGrid:
    @staticmethod
    def test_make_grid_static():
        # arrange
        grid = (100, 1000)
        assert grid[0] != meta[META_N_OUTER]
        assert grid[0] != meta[META_N_INNER]

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
        assert (meta[META_N_OUTER], meta[META_N_MID3D], meta[META_N_INNER]) == grid_fun(meta)

    @staticmethod
    @pytest.mark.parametrize("n", (3, 30, 300))
    @pytest.mark.parametrize("n_threads", (1, 2, 3))
    def test_make_irng_static(n, n_threads):
        # arrange
        assert n != meta[META_N_OUTER]

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
            assert subdomain(meta[META_N_OUTER], thread_id, n_threads) == irng_fun(meta, thread_id)


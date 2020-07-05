"""
Created at 20.03.2020
"""

from .static_grid import make_chunk, make_domain
from .traversals_impl_scalar import _make_apply_scalar, _make_fill_halos_scalar
from .traversals_impl_vector import _make_apply_vector, _make_fill_halos_vector


class Traversals:
    def __init__(self, grid, halo, jit_flags, n_threads):
        assert not (n_threads > 1 and len(grid) == 1)
        domain = make_domain((grid[0], grid[1] if len(grid) > 1 else 0))
        self.n_dims = len(grid)
        chunk = make_chunk(grid[0], n_threads)
        self._fill_halos_scalar = _make_fill_halos_scalar(
            jit_flags=jit_flags, halo=halo, n_dims=self.n_dims, chunk=chunk, domain=domain)
        self._fill_halos_vector = _make_fill_halos_vector(
            jit_flags=jit_flags, halo=halo, n_dims=self.n_dims, chunk=chunk, domain=domain)
        self._apply_scalar = _make_apply_scalar(loop=False, jit_flags=jit_flags, n_dims=self.n_dims,
                                                halo=halo, n_threads=n_threads,
                                                chunk=chunk, domain=domain,
                                                boundary_cond_vector=self._fill_halos_vector,
                                                boundary_cond_scalar=self._fill_halos_scalar)
        self._apply_scalar_loop = _make_apply_scalar(loop=True, jit_flags=jit_flags, n_dims=self.n_dims,
                                                     halo=halo, n_threads=n_threads,
                                                     chunk=chunk, domain=domain,
                                                     boundary_cond_vector=self._fill_halos_vector,
                                                     boundary_cond_scalar=self._fill_halos_scalar
                                                     )
        self._apply_vector = _make_apply_vector(jit_flags=jit_flags, halo=halo, n_dims=self.n_dims,
                                                n_threads=n_threads, domain=domain, chunk=chunk,
                                                boundary_cond_vector=self._fill_halos_vector,
                                                boundary_cond_scalar=self._fill_halos_scalar)

    def apply_scalar(self, *, loop):
        if loop:
            return self._apply_scalar_loop
        else:
            return self._apply_scalar

    def apply_vector(self):
        return self._apply_vector

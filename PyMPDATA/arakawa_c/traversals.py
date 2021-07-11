from .grid import make_chunk, make_domain
from .traversals_impl_scalar import _make_apply_scalar, _make_fill_halos_scalar
from .traversals_impl_vector import _make_apply_vector, _make_fill_halos_vector
from .enumerations import INNER, MID3D, OUTER
from .scalar_field import ScalarField
from .vector_field import VectorField


class Traversals:
    def __init__(self, grid, halo, jit_flags, n_threads):
        assert not (n_threads > 1 and len(grid) == 1)
        domain = make_domain((
            grid[OUTER] if len(grid) > 1 else 0,
            grid[MID3D] if len(grid) > 2 else 0,
            grid[INNER]
            ))
        self.n_dims = len(grid)
        chunk = make_chunk(grid[OUTER], n_threads)
        self._fill_halos_scalar = _make_fill_halos_scalar(
            jit_flags=jit_flags, halo=halo, n_dims=self.n_dims, chunker=chunk, spanner=domain)
        self._fill_halos_vector = _make_fill_halos_vector(
            jit_flags=jit_flags, halo=halo, n_dims=self.n_dims, chunker=chunk, spanner=domain)
        self._apply_scalar = _make_apply_scalar(loop=False, jit_flags=jit_flags, n_dims=self.n_dims,
                                                halo=halo, n_threads=n_threads,
                                                chunker=chunk, spanner=domain,
                                                boundary_cond_vector=self._fill_halos_vector,
                                                boundary_cond_scalar=self._fill_halos_scalar)
        self._apply_scalar_loop = _make_apply_scalar(loop=True, jit_flags=jit_flags, n_dims=self.n_dims,
                                                     halo=halo, n_threads=n_threads,
                                                     chunker=chunk, spanner=domain,
                                                     boundary_cond_vector=self._fill_halos_vector,
                                                     boundary_cond_scalar=self._fill_halos_scalar
                                                     )
        self._apply_vector = _make_apply_vector(jit_flags=jit_flags, halo=halo, n_dims=self.n_dims,
                                                n_threads=n_threads, spanner=domain, chunker=chunk,
                                                boundary_cond_vector=self._fill_halos_vector,
                                                boundary_cond_scalar=self._fill_halos_scalar)
        self.null_scalar_field = ScalarField.make_null(self.n_dims)
        self.null_vector_field = VectorField.make_null(self.n_dims)

    def apply_scalar(self, *, loop):
        if loop:
            return self._apply_scalar_loop
        else:
            return self._apply_scalar

    def apply_vector(self):
        return self._apply_vector

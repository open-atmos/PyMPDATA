""" staggered-grid traversals orchestration """
from ..scalar_field import ScalarField
from ..vector_field import VectorField
from .grid import make_chunk, make_domain
from .indexers import make_indexers
from .traversals_scalar import _make_apply_scalar, _make_fill_halos_scalar
from .traversals_vector import _make_apply_vector, _make_fill_halos_vector
from .enumerations import INNER, MID3D, OUTER


class Traversals:
    """ groups njit-ted traversals for a given grid, halo, jit_flags and threading settings """
    def __init__(self, grid, halo, jit_flags, n_threads):
        assert not (n_threads > 1 and len(grid) == 1)
        domain = make_domain((
            grid[OUTER] if len(grid) > 1 else 0,
            grid[MID3D] if len(grid) > 2 else 0,
            grid[INNER]
            ), jit_flags)
        chunk = make_chunk(grid[OUTER], n_threads, jit_flags)

        self.n_dims = len(grid)
        self.jit_flags = jit_flags
        self.indexers = make_indexers(jit_flags)
        self.null_scalar_field = ScalarField.make_null(self.n_dims, self)
        self.null_vector_field = VectorField.make_null(self.n_dims, self)

        self._code = {}
        self._code['fill_halos_scalar'] = _make_fill_halos_scalar(
            indexers=self.indexers,
            jit_flags=jit_flags, halo=halo, n_dims=self.n_dims,
            chunker=chunk, spanner=domain
        )
        self._code['fill_halos_vector'] = _make_fill_halos_vector(
            indexers=self.indexers,
            jit_flags=jit_flags, halo=halo, n_dims=self.n_dims,
            chunker=chunk, spanner=domain
        )
        self._code['apply_scalar'] = _make_apply_scalar(
            indexers=self.indexers,
            loop=False, jit_flags=jit_flags, n_dims=self.n_dims,
            halo=halo, n_threads=n_threads,
            chunker=chunk, spanner=domain,
            boundary_cond_vector=self._code['fill_halos_vector'],
            boundary_cond_scalar=self._code['fill_halos_scalar']
        )
        self._code['apply_scalar_loop'] = _make_apply_scalar(
            indexers=self.indexers,
            loop=True, jit_flags=jit_flags, n_dims=self.n_dims,
            halo=halo, n_threads=n_threads,
            chunker=chunk, spanner=domain,
            boundary_cond_vector=self._code['fill_halos_vector'],
            boundary_cond_scalar=self._code['fill_halos_scalar']
        )
        self._code['apply_vector'] = _make_apply_vector(
            indexers=self.indexers,
            jit_flags=jit_flags, halo=halo, n_dims=self.n_dims,
            n_threads=n_threads, spanner=domain, chunker=chunk,
            boundary_cond_vector=self._code['fill_halos_vector'],
            boundary_cond_scalar=self._code['fill_halos_scalar']
        )

    def apply_scalar(self, *, loop):
        """ returns scalar field traversal function in two flavours:
            - loop=True sums contributions over dimensions (used in summing upwind fluxes only)
            - loop=False does no summing
        """
        if loop:
            return self._code['apply_scalar_loop']
        return self._code['apply_scalar']

    def apply_vector(self):
        """ returns vector field traversal function """
        return self._code['apply_vector']

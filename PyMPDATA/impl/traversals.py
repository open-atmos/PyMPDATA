""" staggered-grid traversals orchestration """
from collections import namedtuple
from pathlib import Path

import numpy as np  # TODO

from ..scalar_field import ScalarField
from ..vector_field import VectorField
from .enumerations import BUFFER_DEFAULT_VALUE, INNER, MID3D, OUTER
from .grid import make_chunk, make_domain
from .indexers import make_indexers
from .traversals_halos_scalar import _make_fill_halos_scalar
from .traversals_halos_vector import _make_fill_halos_vector
from .traversals_scalar import _make_apply_scalar
from .traversals_vector import _make_apply_vector


class Traversals:
    """groups njit-ted traversals for a given grid, halo, jit_flags and threading settings"""

    def __init__(self, *, grid, halo, jit_flags, n_threads, left_first, buffer_size):
        assert not (n_threads > 1 and len(grid) == 1)
        tmp = (
            grid[OUTER] if len(grid) > 1 else 0,
            grid[MID3D] if len(grid) > 2 else 0,
            grid[INNER],
        )
        domain = make_domain(
            tmp,
            jit_flags,
        )
        chunk = make_chunk(grid[OUTER], n_threads, jit_flags)

        self.n_dims = len(grid)
        self.jit_flags = jit_flags
        self.indexers = make_indexers(jit_flags)

        self.data = namedtuple(  # TODO: rename to data
            Path(__file__).stem + "NullFields", ("scalar", "vector", "buffer")
        )(
            scalar=ScalarField.make_null(self.n_dims, self).impl,  # null_scalar
            vector=VectorField.make_null(self.n_dims, self).impl,  # null_vector
            buffer=np.full((buffer_size,), BUFFER_DEFAULT_VALUE),
        )

        common_kwargs = {
            "jit_flags": jit_flags,
            "halo": halo,
            "n_dims": self.n_dims,
            "chunker": chunk,
            "spanner": domain,
        }
        self._code = {
            "fill_halos_scalar": _make_fill_halos_scalar(
                left_first=left_first,
                **common_kwargs,
            ),
            "fill_halos_vector": _make_fill_halos_vector(
                left_first=left_first,
                **common_kwargs,
            ),
        }
        common_kwargs = {
            **common_kwargs,
            "indexers": self.indexers,
            **{
                "boundary_cond_vector": self._code["fill_halos_vector"],
                "boundary_cond_scalar": self._code["fill_halos_scalar"],
                "n_threads": n_threads,
            },
        }
        self._code = {
            **self._code,
            **{
                "apply_scalar": _make_apply_scalar(
                    loop=False,
                    **common_kwargs,
                ),
                "apply_scalar_loop": _make_apply_scalar(
                    loop=True,
                    **common_kwargs,
                ),
                "apply_vector": _make_apply_vector(
                    **common_kwargs,
                ),
            },
        }

    def apply_scalar(self, *, loop):
        """returns scalar field traversal function in two flavours:
        - loop=True sums contributions over dimensions (used in summing upwind fluxes only)
        - loop=False does no summing
        """
        if loop:
            return self._code["apply_scalar_loop"]
        return self._code["apply_scalar"]

    def apply_vector(self):
        """returns vector field traversal function"""
        return self._code["apply_vector"]

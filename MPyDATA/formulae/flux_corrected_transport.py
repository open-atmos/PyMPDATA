"""
Created at 25.03.2020
"""

import numba
from MPyDATA.arakawa_c.utils import indexers, null_formula
from MPyDATA.jit_flags import jit_flags


def make_psi_extremum(extremum, traversals):
    idx = indexers[traversals.n_dims]
    apply_scalar = traversals.apply_scalar(loop=False)

    formulae = (__make_psi_extremum(traversals.n_dims, idx.at[0], extremum), null_formula)

    @numba.njit(**jit_flags)
    def apply(psi_extremum, psi, psi_bc, null_vecfield, null_vecfield_bc):
        return apply_scalar(*formulae, *psi_extremum, *null_vecfield, *null_vecfield_bc, *psi, *psi_bc)

    return apply


def __make_psi_extremum(n_dims, at, extremum):
    if n_dims == 1:
        @numba.njit(**jit_flags)
        def psi_extremum(_, __, psi):
            return extremum(at(*psi, 0, 0), at(*psi, -1, 0), at(*psi, 1, 0))
    elif n_dims == 2:
        @numba.njit(**jit_flags)
        def psi_extremum(_, __, psi):
            return extremum(at(*psi, 0, 0), at(*psi, -1, 0), at(*psi, 1, 0), at(*psi, 0, -1), at(*psi, 0, 1))
    else:
        raise NotImplementedError()
    return psi_extremum

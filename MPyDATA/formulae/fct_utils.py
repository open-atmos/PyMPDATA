"""
Created at 17.12.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from ..arakawa_c.scalar_field import ScalarField
from ..arakawa_c.vector_field import VectorField

import numpy as np
from ..utils import debug_flag
from .jit_flags import jit_flags

if debug_flag.VALUE:
    import MPyDATA.utils.fake_numba as numba
else:
    import numba

eps = 1e-7


@numba.njit(**jit_flags)
def extremum_3arg(extremum: callable, a1: float, a2: float, a3: float):
    return extremum(extremum(a1, a2), a3)


@numba.njit(**jit_flags)
def extremum_4arg(extremum: callable, a1: float, a2: float, a3: float, a4: float):
    return extremum(extremum(extremum(a1, a2), a3), a4)


@numba.njit(**jit_flags)
def psi_max(psi: ScalarField.Impl):
    a1 = psi.at(-1, 0)
    a2 = psi.at(0, 0)
    a3 = psi.at(1, 0)
    return extremum_3arg(np.maximum, a1, a2, a3)


@numba.njit(**jit_flags)
def psi_min(psi: ScalarField.Impl):
    a1 = psi.at(-1, 0)
    a2 = psi.at(0, 0)
    a3 = psi.at(1, 0)
    return extremum_3arg(np.minimum, a1, a2, a3)


@numba.njit(**jit_flags)
def beta_up(
        psi: ScalarField.Impl,
        psi_max: ScalarField.Impl,
        flx: VectorField.Impl,
        G: ScalarField.Impl
):
    # TODO: loops over dimensions
    assert psi.dimension == 1
    return (
        (
            extremum_4arg(np.maximum, psi_max.at(0, 0), psi.at(-1, 0), psi.at(0, 0), psi.at(1, 0))
            - psi.at(0, 0)
        ) * G.at(0, 0)
    ) / (
        np.maximum(flx.at(-.5, 0), 0)
        - np.minimum(flx.at(+.5, 0), 0)
        + eps
    )


@numba.njit(**jit_flags)
def beta_dn(
        psi: ScalarField.Impl,
        psi_min: ScalarField.Impl,
        flx: VectorField.Impl,
        G: ScalarField.Impl
):
    # TODO: loops over dimensions
    assert psi.dimension == 1
    return (
       (
            psi.at(0, 0)
            - extremum_4arg(np.minimum, psi_min.at(0, 0), psi.at(-1, 0), psi.at(0, 0), psi.at(1, 0))
       ) * G.at(0, 0)
    ) / (
       np.maximum(flx.at(+.5, 0), 0)
       - np.minimum(flx.at(-.5, 0), 0)
       + eps
    )


def make_GC_mono():
    @numba.njit(**jit_flags)
    def fct_GC_mono(
        GC: VectorField.Impl,
        beta_up: ScalarField.Impl,
        beta_dn: ScalarField.Impl
    ):
        # TODO: this version is for iga or positive sign signal only
        result = GC.at(+.5, 0) * np.where(
            # if
            GC.at(+.5, 0) > 0,
            # then
            extremum_3arg(np.minimum, 1, beta_dn.at(0, 0), beta_up.at(1, 0)),
            # else
            extremum_3arg(np.minimum, 1, beta_up.at(0, 0), beta_dn.at(1, 0))
        )
        return result
    return fct_GC_mono
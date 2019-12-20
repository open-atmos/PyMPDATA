"""
Created at 17.12.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from MPyDATA.fields import scalar_field, vector_field
import numpy as np
from MPyDATA_tests.utils import debug

if debug.DEBUG:
    import MPyDATA_tests.utils.fake_numba as numba
else:
    import numba

eps = 1e-7


@numba.njit
def extremum_3arg(extremum: callable, a1: float, a2: float, a3: float):
    return extremum(extremum(a1, a2), a3)


@numba.njit
def extremum_4arg(extremum: callable, a1: float, a2: float, a3: float, a4: float):
    return extremum(extremum(extremum(a1, a2), a3), a4)


@numba.njit
def psi_max(psi: scalar_field.Interface):
    a1 = psi.at(-1, 0)
    a2 = psi.at(0, 0)
    a3 = psi.at(1, 0)
    return extremum_3arg(np.maximum, a1, a2, a3)


@numba.njit
def psi_min(psi: scalar_field.Interface):
    a1 = psi.at(-1, 0)
    a2 = psi.at(0, 0)
    a3 = psi.at(1, 0)
    return extremum_3arg(np.minimum, a1, a2, a3)


def beta_up(
        psi: scalar_field.Interface,
        psi_max: scalar_field.Interface,
        flx: vector_field.Interface,
        G: scalar_field.Interface
):
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


def beta_dn(
        psi: scalar_field.Interface,
        psi_min: scalar_field.Interface,
        flx: vector_field.Interface,
        G: scalar_field.Interface
):
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


#
# def fct_GC_mono(GC, beta_up: scalar_field.Interface, beta_dn: scalar_field.Interface):
#
#     result = GC.at(+.5, 0) * np.where(
#         # if
#         GC.at(+.5, 0) > 0,
#         # then
#         fct_extremum(np.minimum,
#                        1,
#                        beta_dn.at(0,0),
#                        beta_up.at(1,0)
#                        ),
#         # else
#         fct_extremum(np.minimum,
#                        1,
#                        beta_up.at(0,0),
#                        beta_dn.at(1,0)
#                        )
#     )
#
#     return result


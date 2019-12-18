"""
Created at 17.12.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from MPyDATA.fields.scalar_field import ScalarField
from MPyDATA.fields.vector_field import VectorField
from MPyDATA.opts import Opts
import numpy as np
import numba



def make_extremum():
    @numba.njit
    def fct_extremum(extremum, a1, a2, a3, a4=None):
        if a4 is None:
            return extremum(extremum(a1, a2), a3)
        return extremum(extremum(extremum(a1, a2), a3), a4)

    @numba.njit
    def fct_running_extremum(psi: ScalarField, extremum: callable):
        a1 = psi.at(-1, 0)
        a2 = psi.at(0, 0)
        a3 = psi.at(1, 0)

        return fct_extremum(extremum, a1, a2, a3)

    @numba.njit
    def fct_running_maximum(psi: ScalarField):
        return fct_running_extremum(psi, np.maximum)

    @numba.njit
    def fct_running_minimum(psi: ScalarField):
        return fct_running_extremum(psi, np.minimum)

    return fct_extremum

def make_betas(opts: Opts):
    eps = opts.eps

    def fct_beta_up(psi: ScalarField, psi_max: ScalarField, flx: VectorField, G: ScalarField):
        return (
                       (fct_extremum(np.maximum, psi_max.at(0,0), psi.at(-1,0), psi.at(0,0), psi.at(1,0)) - psi.at(0,0)) * G.at(0,0)
               ) / (
                       np.maximum(flx.at(-.5, 0), 0)
                       - np.minimum(flx.at(+.5, 0), 0)
                       + eps
               )


    def fct_beta_dn(psi: ScalarField, psi_min: ScalarField, flx: VectorField, G: ScalarField):
        return (
                       (psi.at(0, 0) - fct_extremum(np.minimum, psi_min.at(0, 0), psi.at(-1, 0), psi.at(0, 0), psi.at(1, 0))) * G.at(0, 0)
               ) / (
                       np.maximum(flx.at(+.5, 0), 0)
                       - np.minimum(flx.at(-.5, 0), 0)
                       + eps
               )
    return fct_beta_up, fct_beta_dn

def make_fct_GC_mono(opts):
    def fct_GC_mono(GC: VectorField, beta_up: ScalarField, beta_dn: ScalarField):

        result = GC.at(+.5, 0) * np.where(
            # if
            GC.at(+.5, 0) > 0,
            # then
            fct_extremum(np.minimum,
                           1,
                           beta_dn.at(0,0),
                           beta_up.at(1,0)
                           ),
            # else
            fct_extremum(np.minimum,
                           1,
                           beta_up.at(0,0),
                           beta_dn.at(1,0)
                           )
        )

        return result
"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from .arakawa_c.scalar_field import ScalarField
from .arakawa_c.vector_field import VectorField
import numpy as np


class MPDATA:
    def __init__(self, step_impl, advectee: ScalarField, advector: VectorField,
                 g_factor: [ScalarField, None] = None):
        self.step_impl = step_impl
        self.curr = advectee
        self.GC_phys = advector
        self.g_factor = g_factor if g_factor is not None else ScalarField.make_null(advectee.n_dims)

        self._vectmp_a = VectorField.clone(advector)
        self._vectmp_b = VectorField.clone(advector)
        self._vectmp_c = VectorField.clone(advector) # TODO: only for mu_coeff != 0

    def step(self, nt):
        self.step_impl(nt,
                       *self.curr.impl,
                       *self.GC_phys.impl,
                       *self.g_factor.impl,
                       *self._vectmp_a.impl,
                       *self._vectmp_b.impl,
                       *self._vectmp_c.impl
                       )


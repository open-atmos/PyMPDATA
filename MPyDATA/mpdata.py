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
        self.flux = VectorField.clone(advector)
        self.GC_phys = advector
        self.g_factor_impl = g_factor.data if g_factor is not None else np.empty([0] * advector.n_dims)
        self.GC_anti = VectorField.clone(advector)

    def step(self, nt):
        self.step_impl(nt, self.curr.impl, self.flux.impl, self.GC_phys.impl, self.GC_anti.impl, self.g_factor_impl)


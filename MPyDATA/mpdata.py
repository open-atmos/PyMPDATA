"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from .arakawa_c.vector_field import VectorField
import numpy as np


class MPDATA:
    def __init__(self,
                 step_impl,
                 advectee,
                 advector,
                 g_factor=None
                 ):
        self.step_impl = step_impl
        self.curr = advectee
        self.flux = VectorField.clone(advector)
        self.GC = advector
        self.g_factor = g_factor if g_factor is not None else np.empty([0] * advector.n_dims)

    def step(self, nt, debug: bool=False):
        n_dims = self.GC.n_dims

        psi = self.curr.data
        flux_0 = self.flux.data[0]
        flux_1 = self.flux.data[1] if n_dims > 1 else np.empty(0, dtype=flux_0.dtype)
        GC_phys_0 = self.GC.data[0]
        GC_phys_1 = self.GC.data[1] if n_dims > 1 else np.empty(0, dtype=flux_1.dtype)
        g_factor = self.g_factor.data

        self.step_impl(nt, psi, flux_0, flux_1, GC_phys_0, GC_phys_1, g_factor)

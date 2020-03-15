"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from .arrays import Arrays
import numpy as np


class MPDATA:
    def __init__(self,
                 step_impl,
                 advectee,
                 advector,
                 g_factor=None
                 ):
        self.step_impl = step_impl
        self.arrays = Arrays(advectee, advector, g_factor)

    def step(self, nt, debug: bool=False):
        n_dims = self.arrays.GC.n_dims

        psi = self.arrays.curr.data
        flux_0 = self.arrays.flux.data[0]
        flux_1 = self.arrays.flux.data[1] if n_dims > 1 else np.empty(0, dtype=flux_0.dtype)
        GC_phys_0 = self.arrays.GC.data[0]
        GC_phys_1 = self.arrays.GC.data[1] if n_dims > 1 else np.empty(0, dtype=flux_1.dtype)
        g_factor = self.arrays.g_factor.data

        self.step_impl(nt, psi, flux_0, flux_1, GC_phys_0, GC_phys_1, g_factor)

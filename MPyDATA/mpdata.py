"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from .arrays import Arrays


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
        psi = self.arrays.curr.data
        flux_0 = self.arrays.flux.data_0
        flux_1 = self.arrays.flux.data_1
        GC_phys_0 = self.arrays.GC.data_0
        GC_phys_1 = self.arrays.GC.data_1
        g_factor = self.arrays.g_factor.data

        self.step_impl(nt, psi, flux_0, flux_1, GC_phys_0, GC_phys_1, g_factor)

"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from .arakawa_c.scalar_field import ScalarField
from .arakawa_c.vector_field import VectorField


class MPDATA:
    def __init__(self, options, step_impl, advectee: ScalarField, advector: VectorField,
                 g_factor: [ScalarField, None] = None):
        self.step_impl = step_impl
        self.curr = advectee
        self.GC_phys = advector
        self.g_factor = g_factor if g_factor is not None else ScalarField.make_null(advectee.n_dims)

        self._vectmp_a = VectorField.clone(advector)
        self._vectmp_b = VectorField.clone(advector)
        if options.mu_coeff != 0:
            self._vectmp_c = VectorField.clone(advector)
        else:
            self._vectmp_c = VectorField.make_null(advector.n_dims)
        if options.flux_corrected_transport:
            self.advectee_min = ScalarField.clone(advectee)
            self.advectee_max = ScalarField.clone(advectee)
        else:
            self.advectee_min = ScalarField.make_null(advectee.n_dims)
            self.advectee_max = ScalarField.make_null(advectee.n_dims)

    def step(self, nt):
        self.step_impl(nt,
                       *self.curr.impl,
                       *self.GC_phys.impl,
                       *self.g_factor.impl,
                       *self._vectmp_a.impl,
                       *self._vectmp_b.impl,
                       *self._vectmp_c.impl,
                       *self.advectee_min.impl,
                       *self.advectee_max.impl
                       )


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
        scalar_field = lambda: ScalarField.clone(advectee)
        null_scalar_field = lambda: ScalarField.make_null(advectee.n_dims)
        vector_field = lambda: VectorField.clone(advector)
        null_vector_field = lambda: VectorField.make_null(advector.n_dims)

        self.step_impl = step_impl
        self.curr = advectee
        self.GC_phys = advector
        self.g_factor = g_factor or null_scalar_field()
        self._vectmp_a = vector_field()
        self._vectmp_b = vector_field()
        self._vectmp_c = vector_field() if options.mu_coeff != 0 else null_vector_field()
        fct = options.flux_corrected_transport
        self.advectee_min = scalar_field() if fct else null_scalar_field()
        self.advectee_max = scalar_field() if fct else null_scalar_field()
        self.beta_up = scalar_field() if fct else null_scalar_field()
        self.beta_down = scalar_field() if fct else null_scalar_field()

    def step(self, nt):
        self.step_impl(nt,
                       *self.curr.impl,
                       *self.GC_phys.impl,
                       *self.g_factor.impl,
                       *self._vectmp_a.impl,
                       *self._vectmp_b.impl,
                       *self._vectmp_c.impl,
                       *self.advectee_min.impl,
                       *self.advectee_max.impl,
                       *self.beta_up.impl,
                       *self.beta_down.impl
                       )

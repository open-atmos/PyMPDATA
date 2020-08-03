"""
Created at 25.09.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from .arakawa_c.scalar_field import ScalarField
from .arakawa_c.vector_field import VectorField
from .stepper import Stepper


class Solver:
    def __init__(self, stepper: Stepper, advectee: ScalarField, advector: VectorField,
                 g_factor: [ScalarField, None] = None):
        scalar_field = lambda: ScalarField.clone(advectee)
        null_scalar_field = lambda: ScalarField.make_null(advectee.n_dims)
        vector_field = lambda: VectorField.clone(advector)
        null_vector_field = lambda: VectorField.make_null(advector.n_dims)

        self.options = stepper.options

        for field in [advector, advectee] + ([g_factor] if g_factor is not None else []):
            assert field.halo == self.options.n_halo
            assert field.dtype == self.options.dtype

        self.stepper = stepper
        self.advectee = advectee
        self.advector = advector
        self.g_factor = g_factor or null_scalar_field()
        self._vectmp_a = vector_field()
        self._vectmp_b = vector_field()
        self._vectmp_c = vector_field() if self.options.non_zero_mu_coeff else null_vector_field()
        fct = self.options.flux_corrected_transport
        self.advectee_min = scalar_field() if fct else null_scalar_field()
        self.advectee_max = scalar_field() if fct else null_scalar_field()
        self.beta_up = scalar_field() if fct else null_scalar_field()
        self.beta_down = scalar_field() if fct else null_scalar_field()

    def advance(self, nt: int, mu_coeff: float = 0):
        assert mu_coeff == 0 or self.options.non_zero_mu_coeff
        wall_time_per_timestep = self.stepper(nt, mu_coeff,
                                              *self.advectee.impl,
                                              *self.advector.impl,
                                              *self.g_factor.impl,
                                              *self._vectmp_a.impl,
                                              *self._vectmp_b.impl,
                                              *self._vectmp_c.impl,
                                              *self.advectee_min.impl,
                                              *self.advectee_max.impl,
                                              *self.beta_up.impl,
                                              *self.beta_down.impl
                                              )
        return wall_time_per_timestep

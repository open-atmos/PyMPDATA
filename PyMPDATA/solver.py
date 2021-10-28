"""
class grouping user-supplied stepper, fields and post-step/post-iter hooks,
as well as self-initialised temporary storage
"""
from typing import Union
import numba
from .scalar_field import  ScalarField
from .vector_field import VectorField
from .stepper import Stepper
from .impl.meta import META_IS_NULL


@numba.njit(inline='always')
# pylint: disable-next=unused-argument
def post_step_null(psi, t):
    pass


@numba.experimental.jitclass([])
class PostIterNull:
    def __init__(self):
        pass

    def __call__(self, flux, g_factor, step, iteration):  # pylint: disable=unused-argument
        pass


class Solver:
    def __init__(self, stepper: Stepper, advectee: ScalarField, advector: VectorField,
                 g_factor: [ScalarField, None] = None):
        scalar_field = lambda dtype=None: ScalarField.clone(advectee, dtype=dtype)
        null_scalar_field = lambda: ScalarField.make_null(advectee.n_dims, stepper.traversals)
        vector_field = lambda: VectorField.clone(advector)
        null_vector_field = lambda: VectorField.make_null(advector.n_dims, stepper.traversals)

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
        self._vectmp_c = vector_field() \
            if self.options.non_zero_mu_coeff else null_vector_field()
        self.nonosc_xtrm = scalar_field(dtype=complex) \
            if self.options.nonoscillatory else null_scalar_field()
        self.nonosc_beta = scalar_field(dtype=complex) \
            if self.options.nonoscillatory else null_scalar_field()

        for field in (self.advectee, self.advector, self.g_factor, self._vectmp_a,
                      self._vectmp_b, self._vectmp_c, self.nonosc_xtrm, self.nonosc_beta):
            field.assemble(self.stepper.traversals)

    def advance(self,
                nt: int,
                mu_coeff: Union[tuple, None] = None,
                post_step=None,
                post_iter=None
                ):
        if mu_coeff is not None:
            assert self.options.non_zero_mu_coeff
        else:
            mu_coeff = (0., 0., 0.)
        if self.options.non_zero_mu_coeff and not self.g_factor.meta[META_IS_NULL]:
            raise NotImplementedError()

        post_step = post_step or post_step_null
        post_iter = post_iter or PostIterNull()

        wall_time_per_timestep = self.stepper(nt, mu_coeff, post_step, post_iter,
                                              *self.advectee.impl,
                                              *self.advector.impl,
                                              *self.g_factor.impl,
                                              *self._vectmp_a.impl,
                                              *self._vectmp_b.impl,
                                              *self._vectmp_c.impl,
                                              *self.nonosc_xtrm.impl,
                                              *self.nonosc_beta.impl
                                              )
        return wall_time_per_timestep

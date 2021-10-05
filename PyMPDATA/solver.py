import numba
from .arakawa_c.scalar_field import ScalarField
from .arakawa_c.vector_field import VectorField
from .stepper import Stepper
from .arakawa_c.meta import META_IS_NULL


@numba.njit()
def post_step_null(psi, t):
    pass


@numba.experimental.jitclass([])
class PostIterNull:
    def __init__(self):
        pass

    def __call__(self, flux, g_factor, t, it):
        pass


class Solver:
    def __init__(self, stepper: Stepper, advectee: ScalarField, advector: VectorField,
                 g_factor: [ScalarField, None] = None):
        scalar_field = lambda dtype=None: ScalarField.clone(advectee, dtype=dtype)
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
        self.nonosc_xtrm = scalar_field(dtype=complex) if fct else null_scalar_field()
        self.nonosc_beta = scalar_field(dtype=complex) if fct else null_scalar_field()

    def advance(self, nt: int, mu_coeff: float = 0., post_step=post_step_null, post_iter=None):
        assert mu_coeff == 0. or self.options.non_zero_mu_coeff
        if mu_coeff != 0 and not self.g_factor.meta[META_IS_NULL]:
            raise NotImplementedError()

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

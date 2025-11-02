"""
class grouping user-supplied stepper, fields and post-step/post-iter hooks,
as well as self-initialised temporary storage
"""

from typing import Hashable, Iterable, Mapping, Union

import numba

from .impl.meta import META_IS_NULL
from .scalar_field import ScalarField
from .stepper import Stepper
from .vector_field import VectorField


@numba.experimental.jitclass([])
class AnteStepNull:  # pylint: disable=too-few-public-methods
    """do-nothing version of the post-step hook"""

    def __init__(self):
        pass

    def call(
        self, advectee, advector, step, index, todo_outer, todo_mid3d, todo_inner
    ):  # pylint: disable-next=unused-argument
        """think of it as a `__call__` method (which Numba does not allow)"""


@numba.experimental.jitclass([])
class PostStepNull:  # pylint: disable=too-few-public-methods
    """do-nothing version of the post-step hook"""

    def __init__(self):
        pass

    def call(self, psi, step, index):  # pylint: disable-next=unused-argument
        """think of it as a `__call__` method (which Numba does not allow)"""


@numba.experimental.jitclass([])
class PostIterNull:  # pylint: disable=too-few-public-methods
    """do-nothing version of the post-iter hook"""

    def __init__(self):
        pass

    def call(self, flux, g_factor, step, iteration):  # pylint: disable=unused-argument
        """think of it as a `__call__` method (which Numba does not allow)"""


class Solver:
    """Solution orchestrator requiring prior instantiation of: a `Stepper`,
    a scalar advectee field (that which is advected),
    a vector advector field (that which advects),
    and optionally a scalar g_factor field (used in some cases of the advection equation).
    Note: in some cases of advection, i.e. momentum advection,
    the advectee can act upon the advector.
    See `PyMPDATA_examples.Jarecka_et_al_2015` for an example of this.
    """

    def __init__(
        self,
        stepper: Stepper,
        advectee: [ScalarField, Iterable[ScalarField], Mapping[Hashable, ScalarField]],
        advector: VectorField,
        g_factor: [ScalarField, None] = None,
    ):
        self.advectee = advectee
        n_dims = advector.n_dims
        if isinstance(advectee, ScalarField):
            self.__fields = {"advectee": (advectee,)}
        #    self.advectee = advectee

        elif isinstance(advectee, Mapping):
            self.__fields = {"advectee": tuple(advectee.values())}
        #    advectee = tuple(advectee.values())
        elif isinstance(advectee, Iterable):
            self.__fields = {"advectee": tuple(advectee)}

        def scalar_field(dtype=None):
            return ScalarField.clone(self.__fields["advectee"][0], dtype=dtype)

        def null_scalar_field():
            return ScalarField.make_null(n_dims, stepper.traversals)

        def vector_field():
            return VectorField.clone(advector)

        def null_vector_field():
            return VectorField.make_null(n_dims, stepper.traversals)

        for field in [advector, *self.__fields["advectee"]] + (
            [g_factor] if g_factor is not None else []
        ):
            assert field.halo == stepper.options.n_halo
            assert field.dtype == stepper.options.dtype
            assert field.grid == advector.grid

        self.__fields.update(
            {
                "advector": advector,
                "g_factor": g_factor or null_scalar_field(),
                "vectmp_a": vector_field(),
                "vectmp_b": vector_field(),
                "vectmp_c": (
                    vector_field()
                    if stepper.options.non_zero_mu_coeff
                    else null_vector_field()
                ),
                "todo_outer": (
                    scalar_field()
                    if stepper.options.dynamic_advector and n_dims > 1
                    else null_scalar_field()
                ),
                "todo_mid3d": (
                    scalar_field()
                    if stepper.options.dynamic_advector and n_dims > 2
                    else null_scalar_field()
                ),
                "todo_inner": (
                    scalar_field()
                    if stepper.options.dynamic_advector
                    else null_scalar_field()
                ),
                "nonosc_xtrm": (
                    scalar_field(dtype=complex)
                    if stepper.options.nonoscillatory
                    else null_scalar_field()
                ),
                "nonosc_beta": (
                    scalar_field(dtype=complex)
                    if stepper.options.nonoscillatory
                    else null_scalar_field()
                ),
            }
        )
        for key, value in self.__fields.items():
            for field in (value,) if key != "advectee" else value:
                field.assemble(stepper.traversals)

        self.__stepper = stepper

    @property
    def advector(self) -> VectorField:
        """advector vector field , todo_outer, todo_mid3d, todo_inner(with halo), unmodified by advance(),
        may be modified from user code"""
        return self.__fields["advector"]

    @property
    def g_factor(self) -> ScalarField:
        """G_factor field (with halo), unmodified by advance(), assumed to be constant-in-time.
        Can be used as a Jacobian for coordinate transformations,
        e.g. into spherical coordinates.
        For this type of usage, see
        `PyMPDATA_examples.Williamson_and_Rasch_1989_as_in_Jaruga_et_al_2015_Fig_14`.
        Can also be used to account for spatial variability of fluid density, see
        `PyMPDATA_examples.Shipway_and_Hill_2012`.
        e.g. the changing density of a fluid."""
        return self.__fields["g_factor"]

    def keys(self):
        return self.key_to_index.keys()

    def advance(
        self,
        n_steps: int,
        mu_coeff: Union[tuple, None] = None,
        ante_step=None,
        post_step=None,
        post_iter=None,
    ):
        """advances solution by `n_steps` steps, optionally accepts: a tuple of diffusion
        coefficients (one value per dimension) as well as `post_iter` and `post_step`
        callbacks expected to be `numba.jitclass`es with a `call` method, for
        signature see `PostStepNull` and `PostIterNull`;
        returns CPU time per timestep (units as returned by `clock.clock()`)"""
        if mu_coeff is not None:
            assert self.__stepper.options.non_zero_mu_coeff
        else:
            mu_coeff = (0.0, 0.0, 0.0)
        if (
            self.__stepper.options.non_zero_mu_coeff
            and not self.__fields["g_factor"].meta[META_IS_NULL]
        ):
            raise NotImplementedError()

        ante_step = ante_step or AnteStepNull()
        post_step = post_step or PostStepNull()
        post_iter = post_iter or PostIterNull()

        return self.__stepper(
            n_steps=n_steps,
            mu_coeff=mu_coeff,
            ante_step=ante_step,
            post_step=post_step,
            post_iter=post_iter,
            fields=self.__fields,
        )

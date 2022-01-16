# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import warnings
from numba.core.errors import NumbaExperimentalFeatureWarning
import numpy as np
from PyMPDATA.impl.formulae_upwind import make_upwind
from PyMPDATA.impl.traversals import Traversals
from PyMPDATA.impl.meta import _Impl
from PyMPDATA.impl.enumerations import IMPL_META_AND_DATA, IMPL_BC
from PyMPDATA import Options, ScalarField, VectorField
from PyMPDATA.boundary_conditions import Periodic


def test_formulae_upwind():
    # Arrange
    psi_data = np.array((0, 1, 0))
    flux_data = np.array((0, 0, 1, 0))

    options = Options()
    halo = options.n_halo
    traversals = Traversals(
        grid=psi_data.shape,
        halo=halo,
        jit_flags=options.jit_flags,
        n_threads=1
    )
    upwind = make_upwind(options=options, non_unit_g_factor=False, traversals=traversals)

    boundary_conditions = (Periodic(),)

    psi = ScalarField(psi_data, halo, boundary_conditions)
    psi.assemble(traversals)
    psi_impl = psi.impl

    flux = VectorField((flux_data,), halo, boundary_conditions)
    flux.assemble(traversals)
    flux_impl = flux.impl

    # Act
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
        upwind(
            traversals.null_impl,
            _Impl(field=psi_impl[IMPL_META_AND_DATA], bc=psi_impl[IMPL_BC]),
            _Impl(field=flux_impl[IMPL_META_AND_DATA], bc=flux_impl[IMPL_BC]),
            _Impl(
               field=traversals.null_impl.scalar[IMPL_META_AND_DATA],
               bc=traversals.null_impl.scalar[IMPL_BC]
            )
        )

    # Assert
    np.testing.assert_array_equal(psi.get(), np.roll(psi_data, 1))

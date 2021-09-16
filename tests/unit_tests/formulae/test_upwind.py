from PyMPDATA.formulae.upwind import make_upwind
from PyMPDATA.arakawa_c.traversals import Traversals
from PyMPDATA import Options, ScalarField, VectorField, PeriodicBoundaryCondition
from numba.core.errors import NumbaExperimentalFeatureWarning
import numpy as np
import warnings


class TestUpwind:
    def test_make_upwind(self):
        # Arrange
        psi_data = np.array((0, 1, 0))
        flux_data = np.array((0, 0, 1, 0))

        options = Options()
        halo = options.n_halo
        traversals = Traversals(grid=psi_data.shape, halo=halo, jit_flags={}, n_threads=1)
        upwind = make_upwind(options=options, non_unit_g_factor=False, traversals=traversals)

        bc = [PeriodicBoundaryCondition()]
        psi = ScalarField(psi_data, halo, bc)
        psi_impl = psi.impl
        flux_impl = VectorField((flux_data,), halo, bc).impl
        null_impl = ScalarField.make_null(len(psi_data.shape)).impl

        # Act
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
            upwind(psi_impl[0], *flux_impl, *null_impl)

        # Assert
        np.testing.assert_array_equal(psi.get(), np.roll(psi_data, 1))

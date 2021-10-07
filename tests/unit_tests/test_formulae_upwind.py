from PyMPDATA.impl.formulae_upwind import make_upwind
from PyMPDATA.impl.traversals import Traversals
from PyMPDATA import Options, ScalarField, VectorField
from PyMPDATA.boundary_conditions import Periodic
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
        traversals = Traversals(grid=psi_data.shape, halo=halo, jit_flags=options.jit_flags, n_threads=1)
        upwind = make_upwind(options=options, non_unit_g_factor=False, traversals=traversals)

        bc = [Periodic()]

        psi = ScalarField(psi_data, halo, bc)
        psi.assemble(traversals)
        psi_impl = psi.impl

        flux = VectorField((flux_data,), halo, bc)
        flux.assemble(traversals)
        flux_impl = flux.impl

        null_impl = ScalarField.make_null(len(psi_data.shape), traversals).impl

        # Act
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=NumbaExperimentalFeatureWarning)
            upwind(psi_impl[0], *flux_impl, *null_impl)

        # Assert
        np.testing.assert_array_equal(psi.get(), np.roll(psi_data, 1))

from MPyDATA.arakawa_c.scalar_field import ScalarField
from MPyDATA.arakawa_c.boundary_conditions.cyclic import CyclicLeft, CyclicRight
import MPyDATA.formulae.fct_utils as fct
import numpy as np


class TestFCTUtils:
    def test_psi_min(self):
        # Arrange
        data = np.array([1, 2, 3])
        psi = ScalarField(data, halo=2, boundary_conditions=((CyclicLeft(), CyclicRight()),))
        psi.fill_halos()
        psi_min = ScalarField.full_like(psi)
        sut = fct.psi_min
        ext=1

        # Act
        psi_min.nd_sum(sut, (psi,), ext=ext)

        # Assert
        np.testing.assert_array_equal(np.array([1]*5), psi_min._impl.data[ext:-ext])

    def test_psi_max(self):
        # Arrange
        data = np.array([1, 2, 3])
        psi = ScalarField(data, halo=2, boundary_conditions=((CyclicLeft(), CyclicRight()),))
        psi_max = ScalarField.full_like(psi)
        psi.fill_halos()
        sut = fct.psi_max
        ext=1

        # Act
        psi_max.nd_sum(sut, (psi,), ext=ext)

        # Assert
        np.testing.assert_array_equal(np.array([3]*5), psi_max._impl.data[ext:-ext])

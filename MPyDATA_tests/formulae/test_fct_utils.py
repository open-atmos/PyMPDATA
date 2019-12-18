from MPyDATA.fields import scalar_field
import MPyDATA.formulae.fct_utils as fct
import numpy as np


class TestFCTUtils:
    def test_psi_min(self):
        # Arrange
        data = np.array([1, 2, 3])
        psi = scalar_field.make(data, halo=2)
        psi.fill_halos()
        psi_min = scalar_field.clone(psi)
        sut = fct.psi_min
        ext=1

        # Act
        scalar_field.apply(sut, psi_min, (psi,), ext=ext)

        # Assert
        np.testing.assert_array_equal(np.array([1]*5), psi_min.data[ext:-ext])

    def test_psi_max(self):
        # Arrange
        data = np.array([1, 2, 3])
        psi = scalar_field.make(data, halo=2)
        psi_max = scalar_field.clone(psi)
        psi.fill_halos()
        sut = fct.psi_max
        ext=1

        # Act
        scalar_field.apply(sut, psi_max, (psi,), ext=ext)

        # Assert
        np.testing.assert_array_equal(np.array([3]*5), psi_max.data[ext:-ext])

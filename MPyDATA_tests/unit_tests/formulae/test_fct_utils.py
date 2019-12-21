from MPyDATA.fields.factories import make_scalar_field, clone_scalar_field
from MPyDATA.fields.utils import apply
import MPyDATA.formulae.fct_utils as fct
import numpy as np


class TestFCTUtils:
    def test_psi_min(self):
        # Arrange
        data = np.array([1, 2, 3])
        psi = make_scalar_field(data, halo=2)
        psi.fill_halos()
        psi_min = clone_scalar_field(psi)
        sut = fct.psi_min
        ext=1

        # Act
        apply(sut, psi_min, (psi,), ext=ext)

        # Assert
        np.testing.assert_array_equal(np.array([1]*5), psi_min._data[ext:-ext])

    def test_psi_max(self):
        # Arrange
        data = np.array([1, 2, 3])
        psi = make_scalar_field(data, halo=2)
        psi_max = clone_scalar_field(psi)
        psi.fill_halos()
        sut = fct.psi_max
        ext=1

        # Act
        apply(sut, psi_max, (psi,), ext=ext)

        # Assert
        np.testing.assert_array_equal(np.array([3]*5), psi_max._data[ext:-ext])

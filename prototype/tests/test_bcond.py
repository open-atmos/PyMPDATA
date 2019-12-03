from MPyDATA import bcond
import numpy as np


def test_bcond_scalar_periodic():
    # Arrange
    opts = {"bcond":'periodic'}
    halo = 2
    psi = np.array([
        0, 0,
        11, 22, 0, 44, 55,
        0, 0
    ])

    nx = len(psi)
    i = slice(halo, nx-halo)

    # Act
    bcond.scalar(opts, psi, i, halo)

    # Assert
    np.testing.assert_array_equal(
        np.array([44, 55, 11, 22, 0, 44, 55, 11, 22]),
        psi
    )

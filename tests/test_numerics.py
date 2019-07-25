"""
Created at 22.07.2019

@author: Michael Olesik
@author: Sylwester Arabas
@author: Piotr Bartman
"""

import numpy as np
from MPyDATA import numerics as nm


def test_flux():
    # Arrange
    psi = np.array([0.25] * 4)
    GC = np.array([0.33, 0.5, 0.66])
    ih = slice(0, 3)
    i = ih + nm.HALF

    # Act
    result = nm.flux(psi, GC, ih)

    # Assert
    print(result)
    assert np.array_equal(result, np.where(GC[ih] >= 0, GC[ih] * psi[i], GC[ih] * psi[i + nm.ONE]))


def test_upwind():
    pass

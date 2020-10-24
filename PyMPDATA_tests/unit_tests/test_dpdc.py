from PyMPDATA.factories import Factories
from PyMPDATA.options import Options
import numpy as np
import pytest


@pytest.mark.parametrize("n_iters", [2, 3, 4])
def test_DPDC(n_iters):
    state = np.array([0, 1, 0])
    C = .5

    mpdata = Factories.constant_1d(state, C, Options(n_iters=n_iters, DPDC=True, flux_corrected_transport=True))
    nt = 1

    conserved = np.sum(mpdata.advectee.get())
    mpdata.advance(nt)
    print(mpdata.advectee.get())

    assert np.sum(mpdata.advectee.get()) == conserved

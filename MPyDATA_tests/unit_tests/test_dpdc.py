from MPyDATA.factories import Factories
from MPyDATA.options import Options
import numpy as np


def test_DPDC():
    state = np.array([0, 1, 0])
    C = .5

    mpdata = Factories.constant_1d(state, C, Options(n_iters=2, DPDC=True))
    nt = 1

    conserved = np.sum(mpdata.advectee.get())
    mpdata.advance(nt)

    assert np.sum(mpdata.advectee.get()) == conserved

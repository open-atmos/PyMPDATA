from MPyDATA.mpdata_factory import MPDATAFactory
from MPyDATA.options import Options
import numpy as np


def test_upwind_1d():
    state = np.array([0, 1, 0])
    C = 1

    mpdata = MPDATAFactory.constant_1d(state, C, Options(n_iters=1))
    nt = 5

    conserved = np.sum(mpdata.curr.get())
    mpdata.step(nt)

    assert np.sum(mpdata.curr.get()) == conserved

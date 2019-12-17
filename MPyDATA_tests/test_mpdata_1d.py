from MPyDATA.mpdata_factory import MPDATAFactory
from MPyDATA.opts import Opts
import numpy as np


def test():
    state = np.array([0,1,0])
    C = 1

    mpdata = MPDATAFactory.uniform_C_1d(state, C, Opts())
    nt = 3

    conserved = np.sum(mpdata.curr.get())
    for _ in range(nt):
        mpdata.step()

    assert np.sum(mpdata.curr.get()) == conserved

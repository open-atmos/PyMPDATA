from MPyDATA.mpdata_factory import MPDATAFactory
#from MPyDATA_examples.Smolarkiewicz_2006_Figs_3_4_10_11_12.setup import Setup

import numpy as np


def test():
#    setup = Setup()

    #dx = (setup.x_max - setup.x_min) / setup.nx
    #xh = np.linspace(setup.x_min, setup.x_max, setup.nx + 1)
    #state = np.diff(setup.cdf_cosine(xh)) / dx

    state = np.array([0,1,0])
    C = 1

    mpdata = MPDATAFactory.uniform_C_1d(state, C, 1)
    nt = 3

    conserved = np.sum(mpdata.curr.get())
    for _ in range(nt):
        # print("\n State before step:\n",mpdata.curr.data)
        mpdata.step()
        # print("\n State after step with velocity {0}:\n".format(str(C)), mpdata.curr.data)

    assert np.sum(mpdata.curr.get()) == conserved

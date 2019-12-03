from MPyDATA.mpdata_factory import MPDATAFactory
from examples.Smolarkiewicz_2006_Figs_3_4_10_11_12 import setup
import numpy as np


class Simulation():
    def __init__(self):
        dx = (setup.x_max - setup.x_min) / setup.nx
        x = np.linspace(setup.x_min+dx/2, setup.x_max-dx/2, setup.nx)
        xh = np.linspace(setup.x_min, setup.x_max, setup.nx+1)
        state = np.diff(setup.cdf_cosine(xh)) / dx

        # TODO: move to smoke tests
        assert x.shape == state.shape
        assert (state >= 0).all()
        import matplotlib.pyplot as plt
        # plt.plot(state)
        # plt.show()

        self.mpdata = MPDATAFactory.uniform_C_1d(state, setup.C)
        self.nt = setup.nt

    def run(self):
        for _ in range(self.nt):
            self.mpdata.step()


if __name__ == '__main__':
    Simulation().run()
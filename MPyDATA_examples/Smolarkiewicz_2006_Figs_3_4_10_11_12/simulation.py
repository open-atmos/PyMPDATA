from MPyDATA.mpdata_factory import MPDATAFactory, from_cdf_1d
from MPyDATA_examples.Smolarkiewicz_2006_Figs_3_4_10_11_12.setup import Setup


class Simulation:
    def __init__(self, setup: Setup):

        x, state = from_cdf_1d(setup.cdf, setup.x_min, setup.x_max, setup.nx)

        self.stepper = MPDATAFactory.constant_1d(
            state,
            setup.C
        )
        self.nt = setup.nt

    @property
    def state(self):
        return self.stepper.curr.get().copy()

    def run(self):
        self.stepper.step(self.nt)

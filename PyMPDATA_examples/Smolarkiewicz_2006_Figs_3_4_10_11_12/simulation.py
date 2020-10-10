from PyMPDATA.factories import Factories
from PyMPDATA.arakawa_c.discretisation import from_cdf_1d
from PyMPDATA_examples.Smolarkiewicz_2006_Figs_3_4_10_11_12.setup import Setup
from PyMPDATA import Options


class Simulation:
    def __init__(self, setup: Setup, options: Options):

        x, state = from_cdf_1d(setup.cdf, setup.x_min, setup.x_max, setup.nx)

        self.stepper = Factories.constant_1d(
            state,
            setup.C,
            options
        )
        self.nt = setup.nt

    @property
    def state(self):
        return self.stepper.advectee.get().copy()

    def run(self):
        self.stepper.advance(self.nt)

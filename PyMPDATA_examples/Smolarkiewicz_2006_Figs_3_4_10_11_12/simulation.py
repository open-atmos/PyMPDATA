from PyMPDATA.factories import Factories
from PyMPDATA.arakawa_c.discretisation import from_cdf_1d
from PyMPDATA_examples.Smolarkiewicz_2006_Figs_3_4_10_11_12.settings import Settings
from PyMPDATA import Options


class Simulation:
    def __init__(self, settings: Settings, options: Options):

        x, state = from_cdf_1d(settings.cdf, settings.x_min, settings.x_max, settings.nx)

        self.stepper = Factories.constant_1d(
            state,
            settings.C,
            options
        )
        self.nt = settings.nt

    @property
    def state(self):
        return self.stepper.advectee.get().copy()

    def run(self):
        self.stepper.advance(self.nt)

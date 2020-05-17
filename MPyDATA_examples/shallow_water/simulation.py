from MPyDATA.factories import Factories
from MPyDATA_examples.shallow_water.setup import Setup
from MPyDATA import Options


class Simulation:
    def __init__(self, setup: Setup, options: Options):
        setup = Setup()
        x = setup.grid
        state = setup.H0(x)

        self.stepper = Factories.shallow_water(
            data=state,
            opts=options)
        self.nt = setup.nt

    @property
    def state(self):
        return self.stepper.curr.get().copy()

    def run(self):
        self.stepper.advance(self.nt)

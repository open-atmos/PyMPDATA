from MPyDATA.mpdata_factory import MPDATAFactory
from MPyDATA.arakawa_c.boundary_conditions.cyclic import CyclicLeft, CyclicRight
from MPyDATA_examples.Smolarkiewicz_2006_Figs_3_4_10_11_12.setup import Setup
import numpy as np
from MPyDATA.options import Options


class Simulation:
    def __init__(self, setup: Setup, opts: Options):
        dx = (setup.x_max - setup.x_min) / setup.nx
        x = np.linspace(setup.x_min+dx/2, setup.x_max-dx/2, setup.nx)
        xh = np.linspace(setup.x_min, setup.x_max, setup.nx+1)
        state = np.diff(setup.cdf(xh)) / dx

        # TODO: move to smoke tests
        assert x.shape == state.shape
        assert (state >= 0).all()

        self.stepper = MPDATAFactory.uniform_C_1d(
            state,
            setup.C,
            opts=opts,
            boundary_conditions=((CyclicLeft(), CyclicRight()),)
        )
        self.nt = setup.nt

    @property
    def state(self):
        return self.stepper.curr.get().copy()

    def run(self, n_iters: int):
        for _ in range(self.nt):
            self.stepper.step(n_iters)

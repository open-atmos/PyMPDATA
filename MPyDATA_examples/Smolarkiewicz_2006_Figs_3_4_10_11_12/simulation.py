from MPyDATA.mpdata_factory import MPDATAFactory
from MPyDATA.arakawa_c.boundary_conditions.cyclic import CyclicLeft, CyclicRight
from MPyDATA_examples.Smolarkiewicz_2006_Figs_3_4_10_11_12.setup import Setup
from MPyDATA.options import Options


class Simulation:
    def __init__(self, setup: Setup, opts: Options, n_iters: int, debug=False):

        x, state = MPDATAFactory.from_cdf_1d(setup.cdf, setup.x_min, setup.x_max, setup.nx)

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
        self.n_iters = n_iters
        self.debug = debug

    @property
    def state(self):
        return self.stepper.arrays.curr.get().copy()

    def run(self):
        for _ in range(self.nt):
            self.stepper.step(self.n_iters, debug=self.debug)

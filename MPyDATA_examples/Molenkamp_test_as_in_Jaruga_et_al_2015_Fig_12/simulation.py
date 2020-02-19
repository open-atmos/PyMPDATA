from MPyDATA.mpdata_factory import MPDATAFactory
from MPyDATA_examples.Molenkamp_test_as_in_Jaruga_et_al_2015_Fig_12.setup import Setup
from MPyDATA.options import Options


class Simulation:
    def __init__(self, setup: Setup, options: Options, n_iters: int, debug=False):

        x, y, z = MPDATAFactory.from_pdf_2d(setup.pdf, xrange=setup.xrange, yrange=setup.yrange, gridsize=setup.grid)
        # TODO: open boundary conditions
        _, self.stepper = MPDATAFactory.kinematic_2d(setup.grid, setup.size, setup.dt, setup.stream_function, field_values={'z': z}, opts=options)
        self.nt = setup.nt
        self.n_iters = n_iters
        self.debug = debug

    @property
    def state(self):
        return self.stepper.mpdatas['z'].arrays.curr.get().copy()

    def run(self):
        for _ in range(self.nt):
            self.stepper.step(self.n_iters, debug=self.debug)

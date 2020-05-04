from MPyDATA import Factories
from MPyDATA_examples.Molenkamp_test_as_in_Jaruga_et_al_2015_Fig_12.setup import Setup
from MPyDATA import Options
from MPyDATA.arakawa_c.discretisation import from_pdf_2d


class Simulation:
    def __init__(self, setup: Setup, options: Options):
        x, y, z = from_pdf_2d(setup.pdf, xrange=setup.xrange, yrange=setup.yrange, gridsize=setup.grid)
        self.mpdata = Factories.stream_function_2d_basic(setup.grid, setup.size, setup.dt, setup.stream_function, z, options)
        self.nt = setup.nt

    @property
    def state(self):
        return self.mpdata.curr.get().copy()

    def run(self):
        self.mpdata.advance(self.nt)

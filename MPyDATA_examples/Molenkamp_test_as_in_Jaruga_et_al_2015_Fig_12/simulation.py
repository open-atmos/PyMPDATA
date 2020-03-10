from MPyDATA.mpdata_factory import MPDATAFactory
from MPyDATA_examples.Molenkamp_test_as_in_Jaruga_et_al_2015_Fig_12.setup import Setup
from MPyDATA.mpdata_factory import from_pdf_2d


class Simulation:
    def __init__(self, setup: Setup):
        x, y, z = from_pdf_2d(setup.pdf, xrange=setup.xrange, yrange=setup.yrange, gridsize=setup.grid)
        self.mpdata = MPDATAFactory.stream_function_2d_basic(setup.grid, setup.size, setup.dt, setup.stream_function, z)
        self.nt = setup.nt

    @property
    def state(self):
        return self.mpdata.arrays.curr.get().copy()

    def run(self):
        self.mpdata.step(self.nt)

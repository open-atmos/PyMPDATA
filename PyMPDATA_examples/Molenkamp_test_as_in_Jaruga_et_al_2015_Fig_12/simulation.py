from PyMPDATA import Factories
from PyMPDATA_examples.Molenkamp_test_as_in_Jaruga_et_al_2015_Fig_12.settings import Settings
from PyMPDATA import Options
from PyMPDATA.arakawa_c.discretisation import from_pdf_2d


class Simulation:
    def __init__(self, settings: Settings, options: Options):
        x, y, z = from_pdf_2d(settings.pdf, xrange=settings.xrange, yrange=settings.yrange, gridsize=settings.grid)
        self.mpdata = Factories.stream_function_2d_basic(settings.grid, settings.size, settings.dt, settings.stream_function, z, options)
        self.nt = settings.nt

    @property
    def state(self):
        return self.mpdata.advectee.get().copy()

    def run(self):
        self.mpdata.advance(self.nt)

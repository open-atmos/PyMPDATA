from PyMPDATA import Options, Stepper, ScalarField, Solver
from PyMPDATA.boundary_conditions import Periodic
from PyMPDATA_examples.utils.discretisation import from_pdf_2d
from PyMPDATA_examples.utils import nondivergent_vector_field_2d
from PyMPDATA_examples.Molenkamp_test_as_in_Jaruga_et_al_2015_Fig_12.settings import Settings


class Simulation:
    def __init__(self, settings: Settings, options: Options):
        _, __, z = from_pdf_2d(
            settings.pdf,
            xrange=settings.xrange,
            yrange=settings.yrange,
            gridsize=settings.grid
        )
        stepper = Stepper(options=options, grid=settings.grid, non_unit_g_factor=False)
        advector = nondivergent_vector_field_2d(
            settings.grid,
            settings.size,
            settings.dt,
            settings.stream_function,
            options.n_halo
        )
        advectee = ScalarField(z.astype(dtype=options.dtype), halo=options.n_halo,
                               boundary_conditions=(Periodic(), Periodic()))
        self.mpdata = Solver(stepper=stepper, advectee=advectee, advector=advector)
        self.nt = settings.nt

    @property
    def state(self):
        return self.mpdata.advectee.get().copy()

    def run(self):
        self.mpdata.advance(self.nt)

import numpy as np
from PyMPDATA import Options, Solver, Stepper, ScalarField, VectorField
from PyMPDATA.boundary_conditions import Periodic
from PyMPDATA_examples.utils.discretisation import from_cdf_1d
from PyMPDATA_examples.Smolarkiewicz_2006_Figs_3_4_10_11_12.settings import Settings


class Simulation:
    def __init__(self, settings: Settings, options: Options):

        _, state = from_cdf_1d(settings.cdf, settings.x_min, settings.x_max, settings.nx)

        self.stepper = Solver(
            stepper=Stepper(options=options, n_dims=len(state.shape), non_unit_g_factor=False),
            advectee=ScalarField(
                state.astype(options.dtype),
                halo=options.n_halo,
                boundary_conditions=(Periodic(),)
            ),
            advector=VectorField(
                (np.full(state.shape[0] + 1, settings.C, dtype=options.dtype),),
                halo=options.n_halo,
                boundary_conditions=(Periodic(),)
            )
        )
        self.nt = settings.nt

    @property
    def state(self):
        return self.stepper.advectee.get().copy()

    def run(self):
        self.stepper.advance(self.nt)

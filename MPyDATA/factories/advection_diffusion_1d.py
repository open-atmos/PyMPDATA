from ..options import Options
from ..mpdata_factory import make_step
from ..arakawa_c.scalar_field import ScalarField
from ..arakawa_c.vector_field import VectorField
from ..mpdata import MPDATA
import numpy as np


def advection_diffusion_1d(options: Options, psi: np.ndarray, C: float, mu: float):
    stepper = make_step(options=options, grid=grid, halo=halo, non_unit_g_factor=False)
    return MPDATA(stepper, advectee=ScalarField(), advector=VectorField(), mu_coeff=mu)

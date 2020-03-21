"""
Created at 21.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np

from .arakawa_c.vector_field import VectorField
from .arakawa_c.scalar_field import ScalarField
from .eulerian_fields import EulerianFields
from .mpdata import MPDATA
from .options import Options
from MPyDATA.factories.step import make_step
from .arakawa_c.discretisation import nondivergent_vector_field_2d


class MPDATAFactory:
    @staticmethod
    def constant_1d(data, C, options: Options):
        halo = 1  # TODO

        mpdata = MPDATA(
            step_impl=make_step(options=options, grid=data.shape, halo=halo, non_unit_g_factor=False),
            advectee=ScalarField(data, halo=halo),
            advector=VectorField((np.full(data.shape[0] + 1, C),), halo=halo)
        )
        return mpdata

    @staticmethod
    def constant_2d(data: np.ndarray, C, options: Options):
        halo = 1  # TODO
        grid = data.shape
        GC_data = [
            np.full((grid[0] + 1, grid[1]), C[0]),
            np.full((grid[0], grid[1] + 1), C[1])
        ]
        GC = VectorField(GC_data, halo=halo)
        state = ScalarField(data=data, halo=halo)
        step = make_step(options=options, grid=grid, halo=halo, non_unit_g_factor=False)
        mpdata = MPDATA(step_impl=step, advectee=state, advector=GC)
        return mpdata

    @staticmethod
    def stream_function_2d_basic(grid, size, dt, stream_function, field, options: Options):
        halo = 1  # TODO
        step = make_step(options=options, grid=grid, halo=halo, non_unit_g_factor=False)
        GC = nondivergent_vector_field_2d(grid, size, dt, stream_function, halo)
        advectee = ScalarField(field, halo=halo)
        return MPDATA(step, advectee=advectee, advector=GC)

    @staticmethod
    def stream_function_2d(grid, size, dt, stream_function, field_values, g_factor, options: Options):
        halo = 1  # TODO
        step = make_step(options=options, grid=grid, halo=halo, non_unit_g_factor=True)
        GC = nondivergent_vector_field_2d(grid, size, dt, stream_function, halo)
        g_factor = ScalarField(g_factor, halo=halo)
        mpdatas = {}
        for k, v in field_values.items():
            advectee = ScalarField(np.full(grid, v), halo=halo)
            mpdatas[k] = MPDATA(step, advectee=advectee, advector=GC, g_factor=g_factor)
        return GC, EulerianFields(mpdatas)

    @staticmethod
    def advection_diffusion_1d(options: Options, psi: np.ndarray, C: float, mu_coeff: float):
        assert psi.ndim == 1
        halo = 1
        grid = psi.shape
        stepper = make_step(options=options, grid=grid, halo=halo, non_unit_g_factor=False, mu_coeff=mu_coeff)
        return MPDATA(stepper,
                      advectee=ScalarField(psi, halo=halo),
                      advector=VectorField((np.full(grid[0]+1, C),), halo=halo)
                      )

"""
Created at 21.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""
from .arakawa_c.vector_field import VectorField
from .arakawa_c.scalar_field import ScalarField
import numpy as np
from .mpdata import MPDATA


class MPDATAFactory:
    @staticmethod
    def kinematic_2d(grid, dt, data):
        GC_data = [
            np.full((grid[0] + 1, grid[1]), -.5),
            np.full((grid[0], grid[1] + 1), .25)
        ]

        GC = VectorField(GC_data[0], GC_data[1])
        state = ScalarField(data=data)
        mpdata = MPDATA(state=state, GC_field=GC)
        return mpdata




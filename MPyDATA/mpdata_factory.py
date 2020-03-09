"""
Created at 21.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

import numpy as np
from .arakawa_c.scalar_field import ScalarField
from .arakawa_c.vector_field import VectorField
from .arakawa_c.boundary_conditions.cyclic import CyclicLeft, CyclicRight
from .mpdata import MPDATA


class MPDATAFactory:
    @staticmethod
    def kinematic_2d(grid, dt, data):
        # TODO
        bcond = (
            (CyclicLeft(), CyclicRight()),
            (CyclicLeft(), CyclicRight())
        )

        halo = 1
        GC = _nondivergent_vector_field_2d(grid, halo, dt, boundary_conditions=bcond)

        state = ScalarField(data=data, halo=halo, boundary_conditions=bcond)
        mpdata = MPDATA(state=state, GC_field=GC)

        return mpdata

    @staticmethod
    def from_pdf_2d(pdf: callable, xrange: list, yrange: list, gridsize: list):
        z = np.empty(gridsize)
        dx, dy = (xrange[1] - xrange[0]) / gridsize[0], (yrange[1] - yrange[0]) / gridsize[1]
        for i in range(gridsize[0]):
            for j in range(gridsize[1]):
                z[i, j] = pdf(
                    xrange[0] + dx*(i+.5),
                    yrange[0] + dy*(j+.5)
                )

        x = np.linspace(xrange[0] + dx / 2, xrange[1] - dx / 2, gridsize[0])
        y = np.linspace(yrange[0] + dy / 2, yrange[1] - dy / 2, gridsize[1])
        return x, y, z


def _nondivergent_vector_field_2d(grid, halo, dt, boundary_conditions):

    GC = [
        np.full((grid[0]+1, grid[1]), .5),
        np.full((grid[0], grid[1]+1), .25)
    ]

    # CFL condition
    for d in range(len(GC)):
        np.testing.assert_array_less(np.abs(GC[d]), 1)

    result = VectorField(data=GC, halo=halo, boundary_conditions=boundary_conditions)

    # nondivergence (of velocity field, hence dt)
    assert np.amax(abs(result.div((dt, dt)).get())) < 5e-9

    return result




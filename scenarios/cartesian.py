""" 2D constant-advector carthesian example """

import numpy as np
from matplotlib import pyplot
from PyMPDATA import ScalarField, Stepper, VectorField
from PyMPDATA.boundary_conditions import Periodic

from PyMPDATA_MPI.domain_decomposition import mpi_indices
from PyMPDATA_MPI.mpi_periodic import MPIPeriodic
from scenarios._scenario import _Scenario


class CartesianScenario(_Scenario):
    """class representation of a test case from
    [Arabas et al. 2014](https://doi.org/10.3233/SPR-140379)
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        mpdata_options,
        n_threads,
        grid,
        rank,
        size,
        courant_field_multiplier,
    ):
        # pylint: disable=too-many-locals, invalid-name
        halo = mpdata_options.n_halo

        xi, yi = mpi_indices(grid, rank, size)
        nx, ny = xi.shape

        boundary_conditions = (MPIPeriodic(size=size), Periodic())
        advectee = ScalarField(
            data=self.initial_condition(xi, yi, grid),
            halo=mpdata_options.n_halo,
            boundary_conditions=boundary_conditions,
        )

        advector = VectorField(
            data=(
                np.full((nx + 1, ny), courant_field_multiplier[0]),
                np.full((nx, ny + 1), courant_field_multiplier[1]),
            ),
            halo=mpdata_options.n_halo,
            boundary_conditions=boundary_conditions,
        )
        stepper = Stepper(
            options=mpdata_options,
            n_dims=2,
            n_threads=n_threads,
            left_first=tuple([rank % 2 == 0] * 2),
            # TODO #70 (see also https://github.com/open-atmos/PyMPDATA/issues/386)
            buffer_size=((ny + 2 * halo) * halo)
            * 2  # for temporary send/recv buffer on one side
            * 2,  # for complex dtype
        )
        super().__init__(stepper=stepper, advectee=advectee, advector=advector)

    @staticmethod
    def initial_condition(xi, yi, grid):
        """returns advectee array for a given grid indices"""
        # pylint: disable=invalid-name
        nx, ny = grid
        x0 = nx / 2
        y0 = ny / 2

        psi = np.exp(
            -((xi + 0.5 - x0) ** 2) / (2 * (nx / 10) ** 2)
            - (yi + 0.5 - y0) ** 2 / (2 * (ny / 10) ** 2)
        )
        return psi

    @staticmethod
    def quick_look(psi, zlim=(-1, 1), norm=None):
        """plots the passed advectee field"""
        # pylint: disable=invalid-name
        xi, yi = np.indices(psi.shape)
        _, ax = pyplot.subplots(subplot_kw={"projection": "3d"})
        pyplot.gca().plot_wireframe(xi + 0.5, yi + 0.5, psi, color="red", linewidth=0.5)
        ax.set_zlim(zlim)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor("black")
            axis.pane.set_alpha(1)
        ax.grid(False)
        ax.set_zticks([])
        ax.set_xlabel("x/dx")
        ax.set_ylabel("y/dy")
        ax.set_proj_type("ortho")
        cnt = ax.contourf(
            xi + 0.5,
            yi + 0.5,
            psi,
            zdir="z",
            offset=-1,
            norm=norm,
            levels=np.linspace(*zlim, 11),
        )
        cbar = pyplot.colorbar(cnt, pad=0.1, aspect=10, fraction=0.04)
        return cbar.norm

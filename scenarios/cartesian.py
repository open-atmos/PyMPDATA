"""2D constant-advector carthesian example"""

import numba
import numpy as np
from matplotlib import pyplot
from PyMPDATA import ScalarField, Stepper, VectorField
from PyMPDATA.boundary_conditions import Periodic
from PyMPDATA.impl.domain_decomposition import make_subdomain
from PyMPDATA.impl.enumerations import INNER, OUTER

from PyMPDATA_MPI.domain_decomposition import mpi_indices
from PyMPDATA_MPI.mpi_periodic import MPIPeriodic
from scenarios._scenario import _Scenario

subdomain = make_subdomain(jit_flags={})


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
        mpi_dim,
    ):
        # pylint: disable=too-many-locals, invalid-name
        halo = mpdata_options.n_halo

        xyi = mpi_indices(grid=grid, rank=rank, size=size, mpi_dim=mpi_dim)
        nx, ny = xyi[mpi_dim].shape

        mpi_periodic = MPIPeriodic(size=size, mpi_dim=mpi_dim)
        periodic = Periodic()
        boundary_conditions = (
            mpi_periodic if mpi_dim == OUTER else periodic,
            mpi_periodic if mpi_dim == INNER else periodic,
        )
        advectee = ScalarField(
            data=self.initial_condition(*xyi, grid),
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
            buffer_size=(
                (ny if mpi_dim == OUTER else nx + 2 * halo) * halo
            )  # TODO #38 support for 3D domain
            * 2  # for temporary send/recv buffer on one side
            * 2  # for complex dtype
            * (2 if mpi_dim == OUTER else n_threads),
        )
        super().__init__(
            mpi_dim=mpi_dim, stepper=stepper, advectee=advectee, advector=advector
        )

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
    def quick_look(psi, n_threads, zlim=(-1, 1), norm=None):
        """plots the passed advectee field"""
        # pylint: disable=invalid-name,too-many-locals
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

        if n_threads > 1 and not numba.config.DISABLE_JIT:  # pylint: disable=no-member
            first_i_with_finite_values = -1
            for i in range(psi.shape[0]):
                if sum(np.isfinite(psi[i, :])) > 0:
                    first_i_with_finite_values = i
            finite_slice = np.isfinite(psi[first_i_with_finite_values, :])
            span = sum(finite_slice)
            assert span != 0
            zero = np.argmax(finite_slice > 0)
            for i in range(n_threads):
                start, stop = subdomain(span, i, n_threads)
                kwargs = {"zs": -1, "zdir": "z", "color": "black", "linestyle": ":"}
                x = [0, psi.shape[0] - 1]
                ax.plot(x, [zero + start] * 2, **kwargs)
                if i == n_threads - 1:
                    ax.plot(x, [zero + stop] * 2, **kwargs)

        cnt = ax.contourf(
            xi + 0.5,
            yi + 0.5,
            psi,
            zdir="z",
            offset=-1,
            norm=norm,
            levels=np.linspace(*zlim, 11),
            alpha=0.75,
        )
        cbar = pyplot.colorbar(cnt, pad=0.1, aspect=10, fraction=0.04)

        return cbar.norm

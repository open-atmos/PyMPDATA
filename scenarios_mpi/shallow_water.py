"""Shallow water equations solver example"""

import numba
import numpy as np
from matplotlib import pyplot
from PyMPDATA_examples.Jarecka_et_al_2015.simulation import make_hooks
from PyMPDATA_MPI.domain_decomposition import mpi_indices
from PyMPDATA_MPI.mpi_periodic import MPIPeriodic

from PyMPDATA import ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Periodic
from PyMPDATA.impl.domain_decomposition import make_subdomain
from PyMPDATA.impl.enumerations import INNER, MAX_DIM_NUM, OUTER
from PyMPDATA.impl.formulae_divide import make_divide_or_zero
from scenarios_mpi._scenario import _Scenario

subdomain = make_subdomain(jit_flags={})


class ShallowWaterScenario(_Scenario):
    # pylint: disable=too-many-instance-attributes, too-many-locals
    """class represenation of shallow water equation solver based on
    [Jarecka_et_al_2015](https://doi.org/10.1016/j.jcp.2015.02.003)."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        mpdata_options,
        n_threads,
        grid,
        rank,
        size,
        courant_field_multiplier,  # pylint: disable=unused-argument
        mpi_dim,
    ):
        def initial_condition(x, y, lx, ly):
            """returns advectee array for a given grid indices"""
            # pylint: disable=invalid-name
            A = 1 / lx / ly
            h = A * (1 - (x / lx) ** 2 - (y / ly) ** 2)
            return np.where(h > 0, h, 0)

        # pylint: disable=invalid-name
        self.halo = mpdata_options.n_halo
        n_threads = n_threads

        xyi = mpi_indices(grid=grid, rank=rank, size=size, mpi_dim=mpi_dim)
        nx, ny = xyi[mpi_dim].shape
        for dim in enumerate(grid):
            xyi[dim[0]] -= (grid[dim[0]] - 1) / 2

        self.rank = rank
        self.dt = 0.1
        self.dx = 32 / grid[0]
        self.dy = 32 / grid[1]
        self.eps = 1e-7
        self.lx0 = 5
        self.ly0 = 2.5

        # pylint: disable=duplicate-code
        mpi_periodic = MPIPeriodic(size=size, mpi_dim=mpi_dim)
        periodic = Periodic()
        boundary_conditions = (
            mpi_periodic if mpi_dim == OUTER else periodic,
            mpi_periodic if mpi_dim == INNER else periodic,
        )
        self.advector = VectorField(
            data=(
                np.zeros((nx + 1, ny)),
                np.zeros((nx, ny + 1)),
            ),
            halo=mpdata_options.n_halo,
            boundary_conditions=boundary_conditions,
        )
        uvgrid = (nx, ny)
        h0 = initial_condition(xyi[0] * self.dx, xyi[1] * self.dy, self.lx0, self.ly0)
        advectees = {
            "h": ScalarField(
                data=h0,
                halo=mpdata_options.n_halo,
                boundary_conditions=boundary_conditions,
            ),
            "uh": ScalarField(
                data=np.zeros(uvgrid),
                halo=mpdata_options.n_halo,
                boundary_conditions=boundary_conditions,
            ),
            "vh": ScalarField(
                data=np.zeros(uvgrid),
                halo=mpdata_options.n_halo,
                boundary_conditions=boundary_conditions,
            ),
        }
        # pylint: disable=duplicate-code
        stepper = Stepper(
            options=mpdata_options,
            n_dims=2,
            n_threads=n_threads,
            left_first=tuple([rank % 2 == 0] * 2),
            # TODO #580 (see also https://github.com/open-atmos/PyMPDATA/issues/386)
            buffer_size=(
                (ny if mpi_dim == OUTER else nx + 2 * self.halo) * self.halo
            )  # TODO #581 support for 3D domain
            * 2  # for temporary send/recv buffer on one side
            * 2  # for complex dtype
            * (2 if mpi_dim == OUTER else n_threads),
        )
        super().__init__(mpi_dim=mpi_dim)
        self.solvers = Solver(stepper, advectees, self.advector)

        self.ante_step, self.post_step = make_hooks(
            traversals=stepper.traversals,
            options=mpdata_options,
            grid_step=(self.dx, None, self.dy),
            time_step=self.dt,
        )

    def __getitem__(self, key):
        return self.solvers.advectee[key].get()

    def data(self, key):
        """Method used to get raw data from advectee"""
        return self.solvers.advectee[key].data

    def _solver_advance(self, n_steps):
        for _ in range(n_steps):
            self.solvers.advance(1, ante_step=self.ante_step, post_step=self.post_step)
        return -1

    @staticmethod
    def quick_look(psi, n_threads, zlim=(-1, 1), norm=None):
        """plots the passed advectee field"""
        # pylint: disable=invalid-name,too-many-locals,duplicate-code
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

"""
2D spherical scenario
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PyMPDATA import ScalarField, Stepper, VectorField

from PyMPDATA_MPI.domain_decomposition import mpi_indices
from PyMPDATA_MPI.mpi_periodic import MPIPeriodic
from PyMPDATA_MPI.mpi_polar import MPIPolar
from scenarios._scenario import _Scenario

# Polar only for upwind: https://github.com/open-atmos/PyMPDATA/issues/120
OPTIONS_KWARGS = ({"n_iters": 1},)


class WilliamsonAndRasch89Settings:
    """Formulae from the paper"""

    # pylint: disable=invalid-name,too-many-instance-attributes,missing-function-docstring
    def __init__(self, *, output_steps, nlon, nlat):
        nt = output_steps[-1]

        self.output_steps = output_steps
        self.nlon = nlon
        self.nlat = nlat

        self.dlmb = 2 * np.pi / nlon
        self.dphi = np.pi / nlat

        self.r = 5 / 64 * np.pi  # original: 7/64*n.pi
        self.x0 = 3 * np.pi / 2
        self.y0 = 0

        self.udt = 2 * np.pi / nt
        self.b = -np.pi / 2
        self.h0 = 0

    def pdf(self, i, j):
        tmp = 2 * (
            (
                np.cos(self.dphi * (j + 0.5) - np.pi / 2)
                * np.sin((self.dlmb * (i + 0.5) - self.x0) / 2)
            )
            ** 2
            + np.sin((self.dphi * (j + 0.5) - np.pi / 2 - self.y0) / 2) ** 2
        )
        return self.h0 + np.where(
            # if
            tmp - self.r**2 <= 0,
            # then
            1 - np.sqrt(tmp) / self.r,
            # else
            0.0,
        )

    def ad_x(self, i, j):
        return (
            self.dlmb
            * self.udt
            * (
                np.cos(self.b) * np.cos(j * self.dphi - np.pi / 2)
                + np.sin(self.b)
                * np.sin(j * self.dphi - np.pi / 2)
                * np.cos((i + 0.5) * self.dlmb)
            )
        )

    def ad_y(self, i, j):
        return (
            -self.dlmb
            * self.udt
            * np.sin(self.b)
            * np.sin(i * self.dlmb)
            * np.cos((j + 0.5) * self.dphi - np.pi / 2)
        )

    def pdf_g_factor(self, _, y):
        return self.dlmb * self.dphi * np.cos(self.dphi * (y + 0.5) - np.pi / 2)


class SphericalScenario(_Scenario):
    """class representation of a test case from
    [Williamson & Rasch 1989](https://doi.org/10.1175/1520-0493(1989)117<0102:TDSLTW>2.0.CO;2)
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        *,
        mpi_dim,
        mpdata_options,
        n_threads,
        grid,
        rank,
        size,
        courant_field_multiplier,
    ):
        # pylint: disable=too-many-locals,invalid-name
        self.settings = WilliamsonAndRasch89Settings(
            nlon=grid[0],  # original: 128
            nlat=grid[1],  # original: 64
            output_steps=range(0, 5120 // 3, 100),  # original: 5120
        )

        xi, _ = mpi_indices(grid=grid, rank=rank, size=size, mpi_dim=mpi_dim)
        mpi_nlon, mpi_nlat = xi.shape

        assert size == 1 or mpi_nlon < self.settings.nlon
        assert mpi_nlat == self.settings.nlat
        x0 = int(xi[0, 0])
        assert x0 == xi[0, 0]

        boundary_conditions = (
            MPIPeriodic(size=size, mpi_dim=mpi_dim),
            MPIPolar(mpi_grid=(mpi_nlon, mpi_nlat), grid=grid, mpi_dim=mpi_dim),
        )

        advector_x = courant_field_multiplier[0] * np.array(
            [
                [self.settings.ad_x(i, j) for j in range(self.settings.nlat)]
                for i in range(x0, x0 + mpi_nlon + 1)
            ]
        )

        advector_y = courant_field_multiplier[1] * np.array(
            [
                [self.settings.ad_y(i, j) for j in range(self.settings.nlat + 1)]
                for i in range(x0, x0 + mpi_nlon)
            ]
        )

        advector = VectorField(
            data=(advector_x, advector_y),
            halo=mpdata_options.n_halo,
            boundary_conditions=boundary_conditions,
        )

        g_factor_z = np.array(
            [
                [self.settings.pdf_g_factor(i, j) for j in range(self.settings.nlat)]
                for i in range(x0, x0 + mpi_nlon)
            ]
        )

        # TODO #81: <move out>
        Cx_max = np.amax(
            np.abs((advector_x[1:, :] + advector_x[:-1, :]) / 2 / g_factor_z)
        )
        assert Cx_max < 1

        Cy_max = np.amax(
            np.abs((advector_y[:, 1:] + advector_y[:, :-1]) / 2 / g_factor_z)
        )
        assert Cy_max < 1
        # TODO #81: </move out>

        g_factor = ScalarField(
            data=g_factor_z,
            halo=mpdata_options.n_halo,
            boundary_conditions=boundary_conditions,
        )

        z = np.array(
            [
                [self.settings.pdf(i, j) for j in range(self.settings.nlat)]
                for i in range(x0, x0 + mpi_nlon)
            ]
        )

        advectee = ScalarField(
            data=z, halo=mpdata_options.n_halo, boundary_conditions=boundary_conditions
        )

        halo = mpdata_options.n_halo
        ny = mpi_nlat
        stepper = Stepper(
            options=mpdata_options,
            non_unit_g_factor=True,
            n_dims=2,
            n_threads=n_threads,
            left_first=tuple([rank % 2 == 0, True]),
            # TODO #70 (see also https://github.com/open-atmos/PyMPDATA/issues/386)
            buffer_size=((ny + 2 * halo) * halo)
            * 2  # for temporary send/recv buffer on one side
            * 2,  # for complex dtype
        )
        super().__init__(
            mpi_dim=mpi_dim,
            stepper=stepper,
            advectee=advectee,
            advector=advector,
            g_factor=g_factor,
        )

    def quick_look(self, state, _):
        """plots the passed advectee field in spherical geometry"""
        # pylint: disable=invalid-name
        theta = np.linspace(0, 1, self.settings.nlat + 1, endpoint=True) * np.pi
        phi = np.linspace(0, 1, self.settings.nlon + 1, endpoint=True) * 2 * np.pi

        XYZ = (
            np.outer(np.sin(theta), np.cos(phi)),
            np.outer(np.sin(theta), np.sin(phi)),
            np.outer(np.cos(theta), np.ones(self.settings.nlon + 1)),
        )
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_axis_off()
        norm = matplotlib.colors.Normalize(
            vmin=self.settings.h0, vmax=self.settings.h0 + 0.05
        )
        ax.plot_surface(
            *XYZ,
            rstride=1,
            cstride=1,
            facecolors=matplotlib.cm.copper_r(  # pylint: disable=no-member
                norm(state.T)
            ),
            alpha=0.6,
            linewidth=0.75,
        )
        m = matplotlib.cm.ScalarMappable(
            cmap=matplotlib.cm.copper_r, norm=norm  # pylint: disable=no-member
        )
        m.set_array([])

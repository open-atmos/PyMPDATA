# pylint: disable=missing-module-docstring,missing-function-docstring,
# pylint: disable=missing-class-docstring,invalid-name,too-many-locals,too-many-arguments,c-extension-no-member
# based on PyMPDATA README example

import mpi4py
import numba_mpi as mpi
import numpy as np
import pytest
from matplotlib import pyplot
from PyMPDATA import Options

from PyMPDATA_MPI.domain_decomposition import subdomain
from PyMPDATA_MPI.hdf_storage import HDFStorage
from PyMPDATA_MPI.settings import Settings
from PyMPDATA_MPI.simulation import Simulation

from .utils import barrier_enclosed, setup_dataset_and_sync_all_workers


class ReadmeSettings(Settings):
    def __init__(self, output_seps):
        self.output_steps = output_seps

    @staticmethod
    def initial_condition(xi, yi, grid):
        nx, ny = grid
        x0 = nx / 2
        y0 = ny / 2

        psi = np.exp(
            -((xi + 0.5 - x0) ** 2) / (2 * (nx / 10) ** 2)
            - (yi + 0.5 - y0) ** 2 / (2 * (ny / 10) ** 2)
        )
        return psi

    @staticmethod
    def quick_look(psi, zlim, norm=None):
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
        cnt = ax.contourf(xi + 0.5, yi + 0.5, psi, zdir="z", offset=-1, norm=norm)
        cbar = pyplot.colorbar(cnt, pad=0.1, aspect=10, fraction=0.04)
        return cbar.norm


@pytest.mark.parametrize(
    "options_kwargs",
    (
        {"n_iters": 1},
        {"n_iters": 2},
        {"n_iters": 2, "third_order_terms": True},
        {"n_iters": 2, "nonoscillatory": True},
        {"n_iters": 3},
    ),
)
@pytest.mark.parametrize("n_threads", (1,))  # TODO #35 : 2+
@pytest.mark.parametrize(
    "output_steps",
    (
        (0,),
        (0, 1),
    ),
)
@pytest.mark.parametrize(
    "courant_field",
    (
        (0.5, 0.25),
        (-0.5, 0.25),
        (0.5, -0.25),
        (-0.5, -0.25),
    ),
)
def test_2d(
    mpi_tmp_path_fixed,
    options_kwargs,
    n_threads,
    output_steps,
    courant_field,
    grid=(24, 24),
    plot=False,
):  # pylint: disable=redefined-outer-name
    options_str = (
        str(options_kwargs)
        .replace(", ", "_")
        .replace(": ", "=")
        .replace("'", "")
        .replace("{", "")
        .replace("}", "")
    )
    paths = {
        mpi_max_size: mpi_tmp_path_fixed
        / f"{options_str}_mpi_max_size_{mpi_max_size}_n_threads_{n_threads}.hdf5"
        for mpi_max_size in (range(1, mpi.size() + 1))
    }

    Storage = HDFStorage
    dataset_name = "test"

    settings = ReadmeSettings(output_steps)

    for mpi_max_size, path in paths.items():
        truncated_size = min(mpi_max_size, mpi.size())
        rank = mpi.rank()

        if rank == 0:
            Storage.create_dataset(
                name=dataset_name, path=path, grid=grid, steps=settings.output_steps
            )

        with Storage.mpi_context(
            path, "r+", mpi4py.MPI.COMM_WORLD.Split(rank < truncated_size, rank)
        ) as storage:
            dataset = setup_dataset_and_sync_all_workers(storage, dataset_name)
            if rank < truncated_size:
                simulation = Simulation(
                    mpdata_options=Options(**options_kwargs),
                    n_threads=n_threads,
                    grid=grid,
                    rank=rank,
                    size=truncated_size,
                    initial_condition=settings.initial_condition,
                    courant_field=courant_field,
                )
                steps_done = 0
                for i, output_step in enumerate(settings.output_steps):
                    n_steps = output_step - steps_done
                    simulation.advance(n_steps=n_steps)
                    steps_done += n_steps

                    x_range = slice(*subdomain(grid[0], rank, truncated_size))
                    dataset[x_range, :, i] = simulation.advectee.get()

    with barrier_enclosed():
        if mpi.rank() != 0:
            with Storage.non_mpi_contex(
                paths[1], "r"
            ) as storage_expected, Storage.non_mpi_contex(
                paths[mpi.rank() + 1], "r"
            ) as storage_actual:
                settings.quick_look(
                    storage_actual[dataset_name][:, :, -1], zlim=(-1, 1)
                )
                if plot:
                    plot_path = f"{options_str}_threads_{n_threads}.pdf"
                    pyplot.savefig(mpi_tmp_path_fixed / plot_path)
                pyplot.close()
                np.testing.assert_array_equal(
                    storage_expected[dataset_name][:, :, -1],
                    storage_actual[dataset_name][:, :, -1],
                )

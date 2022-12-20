# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring,invalid-name,too-many-locals
# based on PyMPDATA README example

from pathlib import Path

import mpi4py
import numba_mpi as mpi
import numpy as np
import pytest

from PySuperDropletLES.hdf_storage import HDFStorage
from PySuperDropletLES.settings import Settings
from PySuperDropletLES.simulation import Simulation

from .fixtures import mpi_tmp_path

assert hasattr(mpi_tmp_path, "_pytestfixturefunction")


@pytest.mark.parametrize("n_iters", (1, 2))
@pytest.mark.parametrize("n_threads", (1,))
def test_2d(
    mpi_tmp_path, n_iters, n_threads, grid=(24, 24)
):  # pylint: disable=redefined-outer-name
    paths = {
        mpi_max_size: Path(mpi_tmp_path)
        / f"n_iters={n_iters}_mpi_max_size_{mpi_max_size}_n_threads_{n_threads}.hdf5"
        for mpi_max_size in range(1, mpi.size() + 1)
    }

    Storage = HDFStorage
    dataset_name = "test"

    settings = Settings()

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

            dataset = storage[dataset_name]
            if rank < truncated_size:
                simulation = Simulation(
                    n_iters=n_iters,
                    n_threads=n_threads,
                    grid=grid,
                    rank=rank,
                    size=truncated_size,
                    initial_condition=settings.initial_condition,
                    courant_field=settings.courant_field,
                )
                steps_done = 0
                for i, output_step in enumerate(settings.output_steps):
                    n_steps = output_step - steps_done
                    simulation.advance(n_steps=n_steps)
                    steps_done += n_steps

                    x_range = slice(
                        *Simulation.subdomain(grid[0], rank, truncated_size)
                    )
                    dataset[x_range, :, i] = simulation.advectee.get()

    mpi.barrier()

    if mpi.rank() != 0:
        with Storage.non_mpi_contex(
            paths[1], "r"
        ) as storage_expected, Storage.non_mpi_contex(
            paths[mpi.rank() + 1], "r"
        ) as storage_actual:
            np.testing.assert_array_equal(
                storage_expected[dataset_name][:, :, :],
                storage_actual[dataset_name][:, :, :],
            )

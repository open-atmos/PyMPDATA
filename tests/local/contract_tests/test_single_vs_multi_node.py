""" test for asserting equivalence of results from single node environment and multi-node one """

import os
import shutil
from pathlib import Path

import numba
import numba_mpi as mpi
import numpy as np
import pytest
from matplotlib import pyplot
from mpi4py import MPI
from PyMPDATA import Options
from PyMPDATA.impl.enumerations import INNER, OUTER

from PyMPDATA_MPI.domain_decomposition import subdomain
from PyMPDATA_MPI.hdf_storage import HDFStorage
from PyMPDATA_MPI.utils import barrier_enclosed, setup_dataset_and_sync_all_workers
from scenarios import CartesianScenario, SphericalScenario

OPTIONS_KWARGS = (
    {"n_iters": 1},
    {"n_iters": 2, "third_order_terms": True},
    {"n_iters": 2, "nonoscillatory": True},
    {"n_iters": 3},
)

COURANT_FIELD_MULTIPLIER = ((0.5, 0.25), (-0.5, 0.25), (0.5, -0.25), (-0.5, -0.25))

CARTESIAN_OUTPUT_STEPS = range(0, 2, 1)

SPHERICAL_OUTPUT_STEPS = range(0, 2000, 100)


@pytest.mark.parametrize(
    "scenario_class, output_steps, n_threads",
    (
        (CartesianScenario, CARTESIAN_OUTPUT_STEPS, 1),
        (CartesianScenario, CARTESIAN_OUTPUT_STEPS, 2),
        (CartesianScenario, CARTESIAN_OUTPUT_STEPS, 3),
        (SphericalScenario, SPHERICAL_OUTPUT_STEPS, 1),  # TODO #56
    ),
)
@pytest.mark.parametrize("options_kwargs", OPTIONS_KWARGS)
@pytest.mark.parametrize("courant_field_multiplier", COURANT_FIELD_MULTIPLIER)
@pytest.mark.parametrize("mpi_dim", (INNER, OUTER))
def test_single_vs_multi_node(  # pylint: disable=too-many-arguments,too-many-branches,too-many-statements
    *,
    mpi_dim,
    scenario_class,
    mpi_tmp_path_fixed,
    options_kwargs,
    n_threads,
    courant_field_multiplier,
    output_steps,
    grid=(64, 32),  # TODO #101
):
    """
    Test is divided into three logical stages.
    During the first stage, root node initializes containers that will store results of computation.
    Second stage performs simulation in a loop over worker pool size.
    Each iteration uses different domain decomposition.
    Last stage is responsible for comparing results to ground truth
    (which is simulation performed on single node environment)
    """
    # pylint: disable=too-many-locals
    if scenario_class is SphericalScenario and options_kwargs["n_iters"] > 1:
        pytest.skip("TODO #56")

    if scenario_class is SphericalScenario and mpi.size() > 2:
        pytest.skip("TODO #56")

    if scenario_class is SphericalScenario and mpi_dim == INNER:
        pytest.skip("TODO #56")

    if n_threads > 1 and options_kwargs.get("nonoscillatory", False):
        pytest.skip("TODO #99")

    if mpi_dim == INNER and options_kwargs.get("third_order_terms", False):
        pytest.skip("TODO #102")

    if n_threads > 1 and numba.config.DISABLE_JIT:  # pylint: disable=no-member
        pytest.skip("threading requires Numba JIT to be enabled")

    plot = True and (
        "CI_PLOTS_PATH" in os.environ
        and courant_field_multiplier == COURANT_FIELD_MULTIPLIER[0]
        and (
            options_kwargs == OPTIONS_KWARGS[-1] or scenario_class is SphericalScenario
        )
    )
    # arrange
    options_str = (
        str(options_kwargs)
        .replace(", ", "_")
        .replace(": ", ".")
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

    # act
    numba.set_num_threads(n_threads)
    for mpi_max_size, path in paths.items():
        truncated_size = min(mpi_max_size, mpi.size())
        rank = mpi.rank()

        courant_str = (
            str(courant_field_multiplier)
            .replace(" ", "")
            .replace(",", ".")
            .replace("(", ".")
            .replace(")", ".")
        )

        plot_path = None
        if plot:
            plot_path = (
                Path(os.environ["CI_PLOTS_PATH"])
                / Path(scenario_class.__name__)
                / Path(
                    f"{options_str}_rank_{mpi.rank()}_size_{truncated_size}_c_field_{courant_str}"
                )
            )
            shutil.rmtree(plot_path, ignore_errors=True)
            os.makedirs(plot_path)
        if rank == 0:
            Storage.create_dataset(
                name=dataset_name, path=path, grid=grid, steps=output_steps
            )

        with Storage.mpi_context(
            path, "r+", MPI.COMM_WORLD.Split(rank < truncated_size, rank)
        ) as storage:
            dataset = setup_dataset_and_sync_all_workers(storage, dataset_name)
            if rank < truncated_size:
                simulation = scenario_class(
                    mpi_dim=mpi_dim,
                    mpdata_options=Options(**options_kwargs),
                    n_threads=n_threads,
                    grid=grid,
                    rank=rank,
                    size=truncated_size,
                    courant_field_multiplier=courant_field_multiplier,
                )
                mpi_range = slice(
                    *subdomain(grid[simulation.mpi_dim], rank, truncated_size)
                )

                simulation.advance(dataset, output_steps, mpi_range)

                # plot
                if plot:
                    tmp = np.empty_like(dataset[:, :, -1])
                    for i, _ in enumerate(output_steps):
                        tmp[:] = np.nan
                        tmp[:, mpi_range] = dataset[:, mpi_range, i]
                        simulation.quick_look(tmp)
                        filename = f"step={i:04d}.svg"
                        pyplot.savefig(plot_path / filename)
                        print("Saving figure")
                        pyplot.close()

    # assert
    with barrier_enclosed():
        path_idx = mpi.rank() + 1
        mode = "r"
        if mpi.rank() != 0:
            with (
                Storage.non_mpi_contex(paths[1], mode) as storage_expected,
                Storage.non_mpi_contex(paths[path_idx], mode) as storage_actual,
            ):
                np.testing.assert_array_equal(
                    storage_expected[dataset_name][:, :, -1],
                    storage_actual[dataset_name][:, :, -1],
                )
        else:
            with Storage.non_mpi_contex(paths[path_idx], mode) as storage_actual:
                no_nans_in_domain = (
                    np.isfinite(storage_actual[dataset_name][:, :, -1])
                ).all()
                assert no_nans_in_domain

                actual = np.asarray(storage_actual[dataset_name])
                if actual.shape[-1] > 1:
                    non_zero_flow = (actual[:, :, 0] != actual[:, :, -1]).any()
                    assert non_zero_flow

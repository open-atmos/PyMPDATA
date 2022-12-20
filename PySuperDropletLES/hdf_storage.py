# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring,invalid-name,too-many-locals

from pathlib import Path

import h5py


class HDFStorage:
    @staticmethod
    def create_dataset(*, name: str, path: Path, grid: tuple, steps: tuple):
        with h5py.File(path, "w") as file:
            file.create_dataset(
                name,
                (*grid, len(steps)),
                dtype="float64",
            )

    @staticmethod
    def mpi_context(path, mode, comm):
        return h5py.File(path, mode, driver="mpio", comm=comm)

    @staticmethod
    def non_mpi_contex(path, mode):
        return h5py.File(path, mode)

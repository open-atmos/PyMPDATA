"""
local pytest customisation
"""

import pathlib
import shutil
import tempfile

import numba_mpi
import pytest
import pytest_mpi


@pytest.fixture(name="mpi_tmp_path_fixed")
def mpi_tmp_path_fixed_fixture():
    """exposes pytest-mpi logic but overrides pytest's temporary file management
    (which tends to fail on cleanup under MPI on macOS)"""

    temp = tempfile.mkdtemp()
    yield pytest_mpi.mpi_tmp_path.__wrapped__(pathlib.Path(temp))
    if numba_mpi.rank() == 0:
        shutil.rmtree(temp)

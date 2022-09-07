# pylint: disable=missing-module-docstring,missing-function-docstring
import numba_mpi
import PySDM
import PyMPDATA


def test_imports():
    for package in [numba_mpi, PySDM, PyMPDATA]:
        print(package.__version__)

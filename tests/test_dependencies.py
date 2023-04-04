# pylint: disable=missing-module-docstring,missing-function-docstring
import numba_mpi
import PyMPDATA


def test_dependencies():
    for package in [numba_mpi, PyMPDATA]:
        print(package.__version__)

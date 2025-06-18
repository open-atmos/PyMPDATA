# pylint: disable=missing-module-docstring,missing-function-docstring
import mpi4py
import numba_mpi
import numpy

import PyMPDATA


def test_dependencies():
    for package in [numba_mpi, PyMPDATA, h5py, mpi4py, numpy]:
        print(package.__version__)

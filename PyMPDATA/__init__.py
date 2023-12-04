"""
Numba-accelerated Pythonic implementation of Multidimensional Positive Definite
Advection Transport Algorithm (MPDATA) with examples in Python, Julia and Matlab

PyMPDATA uses staggered grid with the following node placement for
`PyMPDATA.scalar_field.ScalarField` and
`PyMPDATA.vector_field.VectorField` elements:
![](https://github.com/atmos-cloud-sim-uj/PyMPDATA/releases/download/tip/readme_grid.png)
"""
# pylint: disable=invalid-name
from importlib.metadata import PackageNotFoundError, version

from .options import Options
from .scalar_field import ScalarField
from .solver import Solver
from .stepper import Stepper
from .vector_field import VectorField

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass

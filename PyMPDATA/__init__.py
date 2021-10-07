"""
Numba-accelerated Pythonic implementation of Multidimensional Positive Definite
Advection Transport Algorithm (MPDATA) with examples in Python, Julia and Matlab
"""
# pylint: disable=invalid-name
from pkg_resources import get_distribution, DistributionNotFound, VersionConflict

from .scalar_field import ScalarField
from .vector_field import VectorField
from .options import Options
from .stepper import Stepper
from .solver import Solver

try:
    __version__ = get_distribution(__name__).version
except (DistributionNotFound, VersionConflict):
    # package is not installed
    pass

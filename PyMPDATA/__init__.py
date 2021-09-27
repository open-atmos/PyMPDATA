"""
Numba-accelerated Pythonic implementation of Multidimensional Positive Definite
Advection Transport Algorithm (MPDATA) with examples in Python, Julia and Matlab
"""
# pylint: disable=invalid-name
from pkg_resources import get_distribution, DistributionNotFound, VersionConflict

from .options import Options
from .solver import Solver
from .stepper import Stepper
from .arakawa_c.scalar_field import ScalarField
from .arakawa_c.vector_field import VectorField
from .arakawa_c.boundary_condition import *

try:
    __version__ = get_distribution(__name__).version
except (DistributionNotFound, VersionConflict):
    # package is not installed
    pass

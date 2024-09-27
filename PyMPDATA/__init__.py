"""
.. include:: ../docs/markdown/pympdata_landing.md
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

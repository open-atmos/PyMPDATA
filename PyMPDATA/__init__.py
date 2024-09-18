"""
Numba-accelerated Pythonic implementation of Multidimensional Positive Definite
Advection Transport Algorithm (MPDATA) with examples in Python, Julia and Matlab

PyMPDATA uses staggered grid with the following node placement for
`PyMPDATA.scalar_field.ScalarField` and
`PyMPDATA.vector_field.VectorField` elements:

.. include:: ../docs/markdown/intro.md
.. include:: ../docs/markdown/dependencies.md
.. include:: ../docs/markdown/examples.md
.. include:: ../docs/markdown/options.md
.. include:: ../docs/markdown/grid.md
.. include:: ../docs/markdown/python_grid.md
.. include:: ../docs/markdown/grid_2.md
.. include:: ../docs/markdown/julia_grid.md
.. include:: ../docs/markdown/matlab_grid.md
.. include:: ../docs/markdown/python_grid_2.md
.. include:: ../docs/markdown/grid_3.md
.. include:: ../docs/markdown/stepper_solver.md
.. include:: ../docs/markdown/python_solver.md
.. include:: ../docs/markdown/debugging_contribution.md

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

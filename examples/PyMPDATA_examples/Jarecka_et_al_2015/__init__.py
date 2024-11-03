"""
This module showcases the PyMPDATA implementation of an MPDATA-based shallow-water equations
solver discussed and bencharked against analytical solutions in
[Jarecka_et_al_2015](https://doi.org/10.1016/j.jcp.2015.02.003).

fig_6.ipynb:
.. include:: ./fig_6.ipynb.badges.md
"""

from .plot_output import plot_output
from .settings import Settings
from .simulation import Simulation

"""
This module showcases the PyMPDATA implementation of an MPDATA-based shallow-water equations
solver discussed and bencharked against analytical solutions in
[Jarecka_et_al_2015](https://doi.org/10.1016/j.jcp.2015.02.003).

[![preview notebook](https://img.shields.io/static/v1?label=render%20on&logo=github&color=87ce3e&message=GitHub)](https://github.com/open-atmos/PyMPDATA/blob/main/examples/PyMPDATA_examples/Jarecka_et_al_2015/fig_6.ipynb)
[![launch on mybinder.org](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/open-atmos/PyMPDATA.git/main?urlpath=lab/tree/examples/PyMPDATA_examples/Jarecka_et_al_2015/fig_6.ipynb)
[![launch on Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-atmos/PyMPDATA/blob/main/examples/PyMPDATA_examples/Jarecka_et_al_2015/fig_6.ipynb)
"""

from .plot_output import plot_output
from .settings import Settings
from .simulation import Simulation

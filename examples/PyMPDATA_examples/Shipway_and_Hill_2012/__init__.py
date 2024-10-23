"""
This is an example of 2D droplet size-spectral/spatial problem of
condensational growth in a column of air,
as described in [Shipway and Hill 2012](https://doi.org/10.1002/qj.1913).

[![preview notebook](https://img.shields.io/static/v1?label=render%20on&logo=github&color=87ce3e&message=GitHub)](https://github.com/open-atmos/PyMPDATA/blob/main/examples/PyMPDATA_examples/Shipway_and_Hill_2012/fig_1.ipynb)
[![launch on mybinder.org](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/open-atmos/PyMPDATA.git/main?urlpath=lab/tree/examples/PyMPDATA_examples/Shipway_and_Hill_2012/fig_1.ipynb)
[![launch on Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-atmos/PyMPDATA/blob/main/examples/PyMPDATA_examples/Shipway_and_Hill_2012/fig_1.ipynb)
"""

from .arakawa_c import arakawa_c
from .droplet_activation import DropletActivation
from .formulae import convert_to
from .mpdata import MPDATA
from .plot import plot
from .settings import Settings, const, si

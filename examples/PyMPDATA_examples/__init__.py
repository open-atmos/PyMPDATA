"""
PyMPDATA_examples package includes common Python modules used in PyMPDATA smoke tests
and in example notebooks (but the package wheels do not include the notebooks).
![advection_diffusion_2d](https://github.com/open-atmos/PyMPDATA/releases/download/tip/advection_diffusion.gif)

"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass

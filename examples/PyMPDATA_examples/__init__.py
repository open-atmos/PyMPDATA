"""
.. include:: ../docs/pympdata_examples_landing.md
"""

from importlib.metadata import PackageNotFoundError, version
import PyMPDATA

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass

assert PyMPDATA.__version__ == __version__

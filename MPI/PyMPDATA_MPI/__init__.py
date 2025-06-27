# pylint: disable=invalid-name
"""
.. include:: ../docs/pympdata_mpi_landing.md
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    pass

# pylint: disable=invalid-name
"""
PyMPDATA + numba-mpi coupler sandbox
"""

from pkg_resources import DistributionNotFound, VersionConflict, get_distribution

try:
    __version__ = get_distribution(__name__).version
except (DistributionNotFound, VersionConflict):
    # package is not installed
    pass

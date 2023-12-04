"""
PyMPDATA_examples package includes common Python modules used in PyMPDATA smoke tests
and in example notebooks (but the package wheels do not include the notebooks)
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = get_distribution(__name__).version
except PackageNotFoundError:
    # package is not installed
    pass

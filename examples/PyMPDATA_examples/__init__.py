"""
.. include:: ../docs/pympdata_examples_landing.md
"""

import re
from importlib.metadata import PackageNotFoundError, version

import PyMPDATA

try:
    __version__ = version(__name__)
except PackageNotFoundError:
    # package is not installed
    pass

validator = r"[0-9].+[0-9]"
PyMPDATA_ver = re.search(validator, PyMPDATA.__version__)[0]
examples_ver = re.search(validator, __version__)[0]
assert PyMPDATA_ver == examples_ver

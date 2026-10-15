"""smoke tests for the Burgers' equation numerical simulation."""

from pathlib import Path

import pytest
from open_atmos_jupyter_utils import notebook_vars
from PyMPDATA_examples import burgers_equation

PLOT = False


@pytest.fixture(scope="session", name="variables")
def _variables_fixture():
    return notebook_vars(
        file=Path(burgers_equation.__file__).parent / "burgers_equation.ipynb",
        plot=PLOT,
    )


# TODO #591
# pylint: disable=too-few-public-methods
class TestBurgersEquation:
    """assertions on the final notebook state"""

    pass  # pylint: disable=unnecessary-pass

"""smoke tests for the Burgers' equation numerical simulation."""

from pathlib import Path

import numpy as np
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


class TestBurgersEquation:
    """assertions on the final notebook state"""

    @staticmethod
    def test_vs_analytic(variables):
        pass

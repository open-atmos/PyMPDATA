"""
checking consistency with values in the paper for Figure 19
"""

from pathlib import Path

import numpy as np
import pytest

from PyMPDATA_examples.utils import notebook_vars
from PyMPDATA_examples import Jaruga_et_al_2015



PLOT = False


@pytest.fixture(scope="session", name="variables")
def variables_fixture():
    return notebook_vars(
        file=Path(Jaruga_et_al_2015.__file__).parent / "fig19.ipynb", plot=PLOT
    )


class TestFig19:
    @staticmethod
    def test_range_of_value_at_t0(variables):
        assert np.amin(variables["net"]) == 300
        assert np.amax(variables["net"]) == 300.5


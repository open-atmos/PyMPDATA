"""
tests for Asian (path-dependent) option pricing example using 2D advection-diffusion PDE
"""

from pathlib import Path

import numpy as np
import pytest
from open_atmos_jupyter_utils import notebook_vars
from PyMPDATA_examples import Magnuszewski_et_al_2025

PLOT = False


@pytest.fixture(scope="session", name="variables")
def _variables_fixture():
    return notebook_vars(
        file=Path(Magnuszewski_et_al_2025.__file__).parent
        / "convergence_analysis.ipynb",
        plot=PLOT,
    )


def _datasets(variables):
    return {
        "upwind": variables["l2_errors"]["UPWIND"],
        "mpdata": variables["l2_errors"]["MPDATA"],
        "error_upwind": variables["simulated_errors_upwind"],
        "error_mpdata_2nd": variables["simulated_errors_mpdata_t2"],
        "error_mpdata_3rd": variables["simulated_errors_mpdata_t3"],
    }


class TestFigs:
    """basic assertions for convergence analysis notebook"""

    @staticmethod
    @pytest.mark.parametrize(
        "key",
        ("upwind", "mpdata"),
    )
    def test_convergence_all_converge(variables, key):
        """checks if both MPDATA and UPWIND actually converge"""
        data = _datasets(variables)
        assert (np.diff(data[key]) <= 0).all()

    @staticmethod
    @pytest.mark.parametrize(
        "lower, higher",
        (
            ("error_upwind", "upwind"),
            ("mpdata", "upwind"),
            ("mpdata", "error_mpdata_2nd"),
            ("error_mpdata_3rd", "mpdata"),
        ),
    )
    def test_convergence_order_of_lines(variables, lower, higher):
        """checks if a given set of points is above/below another one"""
        data = _datasets(variables)
        assert data[lower] <= data[higher]

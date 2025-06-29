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
        "mpdata_2it": variables["l2_errors"]["MPDATA (2 it.)"],
        "mpdata_4it": variables["l2_errors"]["MPDATA (4 it.)"],
        "theory_upwind": variables["theory_upwind"],
        "theory_mpdata_2it_t2": variables["theory_mpdata_2it_t2"],
        "theory_mpdata_2it_t3": variables["theory_mpdata_2it_t3"],
        "theory_mpdata_4it_t3": variables["theory_mpdata_4it_t3"],
    }


class TestFigs:
    """basic assertions for convergence analysis notebook"""

    @staticmethod
    @pytest.mark.parametrize(
        "key",
        ("upwind", "mpdata_2it", "mpdata_4it"),
    )
    def test_convergence_all_converge(variables, key):
        """checks if both MPDATA and UPWIND actually converge"""
        data = _datasets(variables)
        assert (np.diff(data[key]) <= 0).all()

    @staticmethod
    @pytest.mark.parametrize(
        "lower, higher",
        (
            ("upwind", "theory_upwind"),
            ("mpdata_2it", "upwind"),
            ("mpdata_4it", "mpdata_2it"),
            ("theory_mpdata_2it_t2", "mpdata_2it"),
        ),
    )
    def test_convergence_order_of_lines(variables, lower, higher):
        """checks if a given set of points is above/below another one"""
        data = _datasets(variables)
        assert data[lower] <= data[higher]

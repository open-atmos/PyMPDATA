"""
tests for Asian (path-dependent) option pricing example using 2D advection-diffusion PDE
"""

from pathlib import Path

import pytest
from open_atmos_jupyter_utils import notebook_vars
from PyMPDATA_examples import Magnuszewski_et_al_2025

PLOT = False


@pytest.fixture(scope="session", name="variables")
def variables_fixture():
    return notebook_vars(
        file=Path(Magnuszewski_et_al_2025.__file__).parent / "figs.ipynb",
        plot=PLOT,
    )


class TestFigs:
    """basic assertions for Fig 1 and Fig 2 data and axes"""

    @staticmethod
    def test_fig_1_axis_ranges(variables):
        """
        checks if both X and Y axes start at -dx/2, -dy/2, respectively """
        for ax in variables["fig1_axs"]:
            assert ax.get_xlim()[0] == -.5
            assert ax.get_ylim()[0] == -.5

    # TODO #543
    @staticmethod
    @pytest.mark.parametrize(
        "condition",
        (
            # UPWIND above MPDATA
            # European analytic above UPWIND
            # MC above geometric Asian analytic and below UPWIND
        ),
    )
    def test_fig_2_order_of_lines(variables, condition):
        """checks if a given set of points is above/below another one"""
        pass

    # TODO #543
    @staticmethod
    def test_fig_2_all_datasets_monotonic(variables):
        """checks if all points within a dataset constitute a monotonically increasing set"""
        pass

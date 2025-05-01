"""
tests for Asian (path-dependent) option pricing example using 2D advection-diffusion PDE
"""

import numpy as np
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

def datasets(variables):
    return {
        'mc': variables['arithmetic_by_mc'],
        'upwind': variables['output']['UPWIND'][-1][:,0],
        'mpdata': variables['output']['MPDATA'][-1][:,0],
        'kemna-vorst': variables['geometric_price'],
        'black-scholes': variables['euro_price'],
    }


class TestFigs:
    """basic assertions for Fig 1 and Fig 2 data and axes"""

    @staticmethod
    @pytest.mark.parametrize("xy, lr, fmt", (
        ("x", 0, "-0.5"),
        ("y", 0, "-0.5"),
        ("x", 1, "{grid_minus_half[0]}"),
        ("y", 1, "{grid_minus_half[1]}"),
    ))
    def test_fig_1_axis_ranges(variables, xy, lr, fmt):
        """
        checks if both X and Y axes start at -dx/2, -dy/2, respectively """
        for axs in variables["fig1_axs"]:
            assert (
                str(getattr(axs, f"get_{xy}lim")()[lr])
                ==
                fmt.format(grid_minus_half=(variables["grid"][0]-.5, variables["grid"][1]-.5))
            )

    @staticmethod
    @pytest.mark.parametrize(
        "lower, higher",
        (
            ("mpdata", "upwind"),
            ("mc", "black-scholes"),# European analytic above UPWIND
            ("kemna-vorst", "mc"),
            ("mc", "upwind"),
        ),
    )
    def test_fig_2_order_of_lines(variables, lower, higher):
        """checks if a given set of points is above/below another one"""
        data = datasets(variables)
        assert (data[lower] <= data[higher]).all()

    @staticmethod
    @pytest.mark.parametrize("key", ('mc', 'upwind', 'mpdata', 'kemna-vorst', 'black-scholes'))
    def test_fig_2_all_datasets_monotonic(variables, key):
        """checks if all points within a dataset constitute a monotonically increasing set"""
        data = datasets(variables)
        assert (np.diff(data[key]) >= 0).all()

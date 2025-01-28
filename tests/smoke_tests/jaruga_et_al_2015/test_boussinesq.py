""" tests for buoyant-bubble test case from Fig. 3 in [Smolarkiewicz & Pudykiewicz
1992](https://doi.org/10.1175/1520-0469(1992)049%3C2082:ACOSLA%3E2.0.CO;2),
as in libmpdata++ paper ([Jaruga et al. 2015](https://doi.org/10.5194/gmd-8-1005-2015), Fig. 19)"""

# pylint: disable=missing-class-docstring,missing-function-docstring

from pathlib import Path

import numpy as np
import pytest
from open_atmos_jupyter_utils import notebook_vars
from PyMPDATA_examples import Jaruga_et_al_2015

PLOT = False


@pytest.fixture(scope="session", name="variables")
def variables_fixture():
    return notebook_vars(
        file=Path(Jaruga_et_al_2015.__file__).parent / "fig19.ipynb",
        plot=PLOT,
    )


class TestFig19:
    @staticmethod
    def test_maximal_theta(variables):
        max_at_t0 = variables["SETUP"].Tht_ref + variables["SETUP"].Tht_dlt
        acceptable_overshoot = 1e-4
        assert (
            max_at_t0 < np.amax(variables["output"]) < max_at_t0 + acceptable_overshoot
        )

    @staticmethod
    def test_minimal_theta(variables):
        min_at_t0 = variables["SETUP"].Tht_ref
        acceptable_undershoot = 1e-4
        assert (
            min_at_t0 - acceptable_undershoot < np.amin(variables["output"]) < min_at_t0
        )

    @staticmethod
    @pytest.mark.parametrize(
        "area",
        (
            (slice(0, 20), slice(None)),
            (slice(80), slice(None)),
            (slice(None), slice(0, 30)),
            (slice(None), slice(70)),
        ),
    )
    def test_theta_at_domain_edges_equal_to_reference_value(area, variables):
        psi_at_last_step = variables["output"][-1, :, :]
        np.testing.assert_approx_equal(
            actual=psi_at_last_step[area],
            desired=variables["SETUP"].Tht_ref,
            significant=2,
        )

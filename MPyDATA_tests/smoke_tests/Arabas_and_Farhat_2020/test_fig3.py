from MPyDATA_examples.Arabas_and_Farhat_2020.analysis_figures_2_and_3 import convergence_in_time
import pytest
import numpy as np


@pytest.fixture(scope="module")
def l2err_vs_l2l2():
    return convergence_in_time(num=2)


class TestFig3:
    @staticmethod
    def test_upwind_below_1st_order(l2err_vs_l2l2):
        for key, value in l2err_vs_l2l2.items():
            if key.startswith("upwind"):
                x, y = value[0], value[1]
                slope = np.diff(y) / np.diff(x)
                np.testing.assert_almost_equal(slope, 1.3, 1)

    @staticmethod
    @pytest.mark.skip() # TODO
    def test_mpdata_above_1st_order(l2err_vs_l2l2):
        for key, value in l2err_vs_l2l2.items():
            if key.startswith("MPDATA"):
                x, y = value[0], value[1]
                slope = np.diff(y) / np.diff(x)
                np.testing.assert_almost_equal(slope, 2.5, 1)

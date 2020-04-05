from MPyDATA_examples.Arabas_and_Farhat_2019.analysis_table_1 import table_1_data
import numpy as np
import pytest


def check(actual, desired):
    assert actual[0] == desired[0]
    assert actual[1] == desired[1]
    assert np.round(actual[2], 1) <= desired[2]
    assert np.round(actual[3], 1) <= desired[3]
    assert np.round(actual[4], 1) <= desired[4]
    np.testing.assert_allclose(actual[5:], desired[5:], rtol=0.0005, atol=0.005)


@pytest.fixture(scope="module")
def tabledata():
    return table_1_data()

@pytest.mark.skip()
def test_row1(tabledata):
    check(actual=tabledata[0], desired=(.25,  80, -8.3, -10.2, -11.3, 19.996, 20.000, 18.089))

@pytest.mark.skip()
def test_row2(tabledata):
    check(actual=tabledata[1], desired=(.25,  90, -8.1, -10.1, -11.3, 10.035, 10.011,  9.045))

@pytest.mark.skip()
def test_row3(tabledata):
    check(actual=tabledata[2], desired=(.25, 100, -8.0, -10.0, -11.3,  3.228,  3.162,  3.037))

@pytest.mark.skip()
def test_row4(tabledata):
    check(actual=tabledata[3], desired=(.25, 110, -8.3, -10.1, -11.3,  0.667, 0.649, 0.640))

@pytest.mark.skip()
def test_row5(tabledata):
    check(actual=tabledata[4], desired=(.25, 120, -8.3, -10.1, -11.3,  0.089, 0.087, 0.086))

@pytest.mark.skip()
def test_row6(tabledata):
    check(actual=tabledata[5], desired=(.5, 80, -8.7, -10.3, -11.4, 19.996, 20.000, 16.648))

@pytest.mark.skip()
def test_row7(tabledata):
    check(actual=tabledata[6], desired=(.5, 90, -8.5, -10.3, -11.4, 10.290, 10.240, 8.834))

@pytest.mark.skip()
def test_row8(tabledata):
    check(actual=tabledata[7], desired=(.5, 100, -8.5, -10.2, -11.3, 4.193, 4.109, 3.785))

@pytest.mark.skip()
def test_row9(tabledata):
    check(actual=tabledata[8], desired=(.5, 110, -8.7, -10.3, -11.4, 1.412, 1.372, 1.312))

@pytest.mark.skip()
def test_row10(tabledata):
    check(actual=tabledata[9], desired=(.5, 120, -8.7, -10.3, -11.4, 0.397, 0.385, 0.376))

@pytest.mark.skip()
def test_row11(tabledata):
    check(actual=tabledata[10], desired=(3, 80, -10.3, -12.2, -13.2, 19.996, 20.000, 10.253))

@pytest.mark.skip()
def test_row12(tabledata):
    check(actual=tabledata[11], desired=(3, 90, -10.3, -12.2, -13.2, 11.696, 11.668, 6.783))

@pytest.mark.skip()
def test_row13(tabledata):
    check(actual=tabledata[12], desired=(3, 100, -10.3, -12.2, -13.2, 6.931, 6.896, 4.406))

@pytest.mark.skip()
def test_row14(tabledata):
    check(actual=tabledata[13], desired=(3, 110, -10.3, -12.2, -13.2, 4.154, 4.118, 2.826))

@pytest.mark.skip()
def test_row15(tabledata):
    check(actual=tabledata[14], desired=(3, 120, -10.3, -12.2, -13.2, 2.510, 2.478, 1.797))

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from PyMPDATA_examples.Molenkamp_test_as_in_Jaruga_et_al_2015_Fig_12.analysis import (
    fig_12_data,
)


@pytest.fixture(scope="module")
def data():
    return fig_12_data()


# pylint: disable-next=redefined-outer-name
def test_upwind(data):
    sut = data[0]
    assert np.amin(sut) == 1
    np.testing.assert_approx_equal(np.amax(sut), 4.796, significant=4)


# pylint: disable-next=redefined-outer-name
def test_2_nonosc(data):
    sut = data[1]
    np.testing.assert_approx_equal(np.amin(sut), 1)
    np.testing.assert_approx_equal(np.amax(sut), 3.52544410, significant=2)


# pylint: disable-next=redefined-outer-name
def test_3_nonosc_tot(data):
    sut = data[2]
    np.testing.assert_approx_equal(np.amin(sut), 1)
    np.testing.assert_approx_equal(np.amax(sut), 4.26672894, significant=2)


# pylint: disable-next=redefined-outer-name
def test_2_nonosc_iga(data):
    sut = data[3]
    np.testing.assert_approx_equal(np.amin(sut), 1)
    np.testing.assert_approx_equal(np.amax(sut), 4.25518091, significant=2)

from MPyDATA_examples.Molenkamp_test_as_in_Jaruga_et_al_2015_Fig_12.analysis import fig_12_data
import pytest
import numpy as np


@pytest.fixture(scope="module")
def data():
    return fig_12_data()


def test_a(data):
    sut = data[0]
    assert np.amin(sut) == 1
    np.testing.assert_approx_equal(np.amax(sut), 4.796, significant=4)


@pytest.mark.skip()
def test_c(data):
    sut = data[2]
    assert np.amin(sut) == 1
    assert np.amax(sut) == 5


@pytest.mark.skip()
def test_d(data):
    sut = data[2]
    assert np.amin(sut) == 1
    assert np.amax(sut) == 5

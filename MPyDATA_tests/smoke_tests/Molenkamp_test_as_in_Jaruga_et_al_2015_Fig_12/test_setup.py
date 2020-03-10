from MPyDATA_examples.Molenkamp_test_as_in_Jaruga_et_al_2015_Fig_12.setup import Setup
from matplotlib import pyplot
from MPyDATA.mpdata_factory import MPDATAFactory
import pytest


@pytest.mark.skip()
def test_pdf(plot=True):
    # Arrange
    setup = Setup()

    # Act
    x, y, z = MPDATAFactory.from_pdf_2d(setup.pdf, xrange=setup.xrange, yrange=setup.yrange, gridsize = setup.grid)
    # Act

    if plot:
        # Plot
        pyplot.imshow(z)
        pyplot.show()

    # Assert

from MPyDATA_examples.Molenkamp_test_as_in_Jaruga_et_al_2015_Fig_12.setup import Setup, h, h0
from matplotlib import pyplot
from MPyDATA.mpdata_factory import from_pdf_2d


def test_pdf(plot=True):
    # Arrange
    setup = Setup()

    # Act
    _, _, z = from_pdf_2d(setup.pdf, xrange=setup.xrange, yrange=setup.yrange, gridsize=setup.grid)
    # Act

    if plot:
        # Plot
        pyplot.imshow(z)
        pyplot.show()

    # Assert
    assert (z >= h0).all()
    assert (z < h0+h).all()

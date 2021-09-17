from PyMPDATA_examples.Molenkamp_test_as_in_Jaruga_et_al_2015_Fig_12.settings import Settings, h, h0
from PyMPDATA.arakawa_c.discretisation import from_pdf_2d
from matplotlib import pyplot


def test_pdf(plot=False):
    # Arrange
    settings = Settings()

    # Act
    _, _, z = from_pdf_2d(settings.pdf, xrange=settings.xrange, yrange=settings.yrange, gridsize=settings.grid)
    # Act

    if plot:
        # Plot
        pyplot.imshow(z)
        pyplot.show()

    # Assert
    assert (z >= h0).all()
    assert (z < h0+h).all()

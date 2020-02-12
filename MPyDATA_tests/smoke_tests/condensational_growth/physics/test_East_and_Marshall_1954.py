from MPyDATA_examples.condensational_growth.physics.East_and_Marshall_1954 import SizeDistribution
import pint
from matplotlib import pyplot
import numpy as np


def diff(x):
    return np.diff(x.magnitude) * x.units


def test_SizeDistribution(plot=True):
    # Arrange
    si = pint.UnitRegistry()
    sd = SizeDistribution(si)

    # Act
    x = np.linspace(1, 18, 100) * si.micrometre
    numpdfx = x[1:] - diff(x) / 2
    numpdfy = diff(sd.cdf(x)) / diff(x)

    # Plot
    if plot:
        # Fig 3 from East & Marshall 1954
        si.setup_matplotlib()
        pyplot.plot(numpdfx, numpdfy, label='cdf')
        pyplot.plot(numpdfx, sd.pdf(numpdfx), label='pdf', linestyle='--')
        pyplot.legend()
        pyplot.gca().yaxis.set_units(1 / si.centimetre ** 3 / si.micrometre)
        pyplot.show()

    # Assert
    relerr = ((sd.pdf(numpdfx) - numpdfy) / numpdfy).magnitude
    assert not (relerr > 0).all()
    assert not (relerr < 0).all()
    assert np.where(
        numpdfy.magnitude < 5,
        True,
        np.abs(relerr) < 1e-2
    ).all()


# TODO
# def test_domain_length():
#     # Arrange
#     si = pint.UnitRegistry()
#     from MPyDATA_examples.condensational_growth.analysis import analysis
#     from MPyDATA_examples.condensational_growth.physics.equilibrium_drop_growth import PdfEvolver
#
#     result, setup = analysis()
#
#     # Act
#     PdfEvolver(setup.pdf, setup.drdt, t)
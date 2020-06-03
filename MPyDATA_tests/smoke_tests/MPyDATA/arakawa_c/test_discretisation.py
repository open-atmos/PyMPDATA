from MPyDATA_examples.Olesik_et_al_2020.physics.East_and_Marshall_1954 import SizeDistribution
from MPyDATA.arakawa_c.discretisation import discretised_analytical_solution
import pint
from matplotlib import pyplot
import numpy as np


def diff(x):
    return np.diff(x.magnitude) * x.units


def test_size_distribution(plot=True):
    # Arrange
    si = pint.UnitRegistry()
    sd = SizeDistribution(si)
    n_unit = si.centimetres ** -3 / si.micrometre
    r_unit = si.micrometre

    # Act
    x = np.linspace(1, 18, 100) * r_unit
    numpdfx = x[1:] - diff(x) / 2
    pdf_t = lambda r:  sd.pdf(r * r_unit).to(n_unit).magnitude
    numpdfy = discretised_analytical_solution(rh=x.magnitude, pdf_t= pdf_t) * n_unit

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

    totalpdf = np.sum(numpdfy * (diff(x)))
    cdf_max = sd.cdf(np.inf * r_unit)
    print(totalpdf, cdf_max, totalpdf-cdf_max)
    from scipy import integrate
    print(integrate.quad(pdf_t, x[0].magnitude, x[-1].magnitude))

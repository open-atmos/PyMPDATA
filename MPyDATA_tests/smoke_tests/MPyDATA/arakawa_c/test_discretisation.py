from MPyDATA_examples.Olesik_et_al_2020.physics.East_and_Marshall_1954 import SizeDistribution
from MPyDATA.arakawa_c.discretisation import discretised_analytical_solution
import pint
from matplotlib import pyplot
import numpy as np
from MPyDATA_examples.Olesik_et_al_2020.coordinates import x_id, x_log_of_pn, x_p2
import pytest
import platform



def diff(x):
    return np.diff(x.magnitude) * x.units

@pytest.mark.parametrize(
    "grid", [x_id(), x_log_of_pn(), x_p2()]
)
@pytest.mark.parametrize(
    "coord", [x_id(), x_log_of_pn(), x_p2()]
)

@pytest.mark.skipif(platform.system() == 'Windows',
                    reason="test is not passing on travis windows build")
def test_size_distribution(grid, coord, plot=True):
    # Arrange
    si = pint.UnitRegistry()
    sd = SizeDistribution(si)
    n_unit = si.centimetres ** -3 / si.micrometre
    r_unit = si.micrometre

    # Act
    x = grid.x(np.linspace(1, 18, 100)) * r_unit
    dx_dr = coord.dx_dr
    numpdfx = x[1:] - diff(x) / 2
    pdf_t = lambda r:  sd.pdf(r * r_unit).to(n_unit).magnitude / dx_dr(r * r_unit).magnitude
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
    totalpdf = np.sum(numpdfy * (diff(x)))
    from scipy import integrate
    integratedpdf, _ = integrate.quad(pdf_t, x[0].magnitude, x[-1].magnitude)
    print(totalpdf, integratedpdf)
    np.testing.assert_array_almost_equal(totalpdf,integratedpdf)


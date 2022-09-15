# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import pytest
from matplotlib import pylab
from PyMPDATA_examples.Jarecka_et_al_2015 import Settings, Simulation, plot_output
from PyMPDATA_examples.utils.error_norms import L2


@pytest.mark.parametrize("n_x", (101, 100))
@pytest.mark.parametrize("n_y", (101, 100))
def test_just_do_it(n_x, n_y, plot=False):
    # arrange
    settings = Settings()
    settings.dx *= settings.nx / n_x
    settings.nx = n_x
    settings.dy *= settings.ny / n_y
    settings.ny = n_y
    simulation = Simulation(settings)
    times = (1, 3, 7)

    # act
    output = simulation.run()

    # plot
    plot_data = plot_output(times, output, settings, return_data=True)
    if plot:
        pylab.show()

    # assert
    for item in plot_data.values():
        assert 2 ** L2(item["h_numeric"], item["h_analytic"], nt=settings.nt) < 5e-3
        assert 2 ** L2(item["q_h_numeric"], item["q_h_analytic"], nt=settings.nt) < 5e-2

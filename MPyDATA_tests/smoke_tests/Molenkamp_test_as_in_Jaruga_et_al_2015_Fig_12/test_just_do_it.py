from MPyDATA_examples.Molenkamp_test_as_in_Jaruga_et_al_2015_Fig_12.simulation import Simulation
from MPyDATA_examples.Molenkamp_test_as_in_Jaruga_et_al_2015_Fig_12.setup import Setup
from MPyDATA.options import Options
import pytest


@pytest.mark.skip
def test_just_do_it():
    opts = Options()
    setup = Setup()
    simulation = Simulation(setup, opts=opts, n_iters=2)
    simulation.run()

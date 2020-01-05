from MPyDATA_examples.Arabas_and_Farhat_2019.simulation import Simulation
from MPyDATA_examples.Arabas_and_Farhat_2019.setup1_european_corridor import Setup


def test_fig1():
    Simulation(Setup()).run(n_iters=2)
    # TODO: assert

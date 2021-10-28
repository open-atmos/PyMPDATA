from PyMPDATA_examples.Arabas_and_Farhat_2020 import Simulation
from PyMPDATA_examples.Arabas_and_Farhat_2020.setup2_american_put import Settings

def test_black_scholes():
    # arrange
    settings = Settings(T=.25, C_opt=.02, S0=80)
    simulation = Simulation(settings)

    # act
    simulation.run(n_iters=2)

    # assert

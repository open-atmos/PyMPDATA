from MPyDATA_examples.condensational_growth.simulation import Simulation
from MPyDATA_examples.condensational_growth.setups import east_1957_fig3
from MPyDATA_examples.condensational_growth.coord import x_id, x_ln, x_p2
from MPyDATA.options import Options



def test_simulation():
    setup = east_1957_fig3.East1957Fig3
    coord = x_id()
    opts = Options(nug=True)

    simulation = Simulation(coord, setup, opts)

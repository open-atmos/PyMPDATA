from MPyDATA_examples.Smolarkiewicz_2006_Figs_3_4_10_11_12.simulation import Simulation
from MPyDATA_examples.Smolarkiewicz_2006_Figs_3_4_10_11_12.setup import Setup
from MPyDATA.options import Options


# TODO: run all cases
# TODO: add asserts (not in "run_all"...)
def test_run_all():
    opts = Options()
    setup = Setup(shape='cosine')
    Simulation(setup, opts).run()

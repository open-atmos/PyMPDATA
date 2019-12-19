from MPyDATA_examples.Smolarkiewicz_2006_Figs_3_4_10_11_12.simulation import Simulation
from MPyDATA_examples.Smolarkiewicz_2006_Figs_3_4_10_11_12.setup import Setup
from MPyDATA.opts import Opts

if __name__ == '__main__':
    opts = Opts()
    setup = Setup(shape='cosine')
    Simulation(setup, opts).run()

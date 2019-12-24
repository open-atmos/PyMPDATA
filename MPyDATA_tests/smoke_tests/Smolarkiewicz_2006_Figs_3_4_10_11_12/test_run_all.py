from MPyDATA_examples.Smolarkiewicz_2006_Figs_3_4_10_11_12.simulation import Simulation
from MPyDATA_examples.Smolarkiewicz_2006_Figs_3_4_10_11_12.setup import Setup
from MPyDATA.options import Options


# TODO: add asserts
class TestSmolarkiewicz_2006:
    @staticmethod
    def test_Fig3():
        Simulation(Setup("cosine"), Options(iga=False, n_iters=1)).run()

    @staticmethod
    def test_Fig4():
        Simulation(Setup("cosine"), Options(iga=False, n_iters=2)).run()

    @staticmethod
    def test_Fig10():
        Simulation(Setup("cosine"), Options(iga=True, n_iters=2)).run()

    @staticmethod
    def test_Fig11():
        Simulation(Setup("rect"), Options(iga=True, n_iters=2)).run()

    @staticmethod
    def test_Fig12():
        Simulation(Setup("rect"), Options(iga=True, fct=True, n_iters=2)).run()



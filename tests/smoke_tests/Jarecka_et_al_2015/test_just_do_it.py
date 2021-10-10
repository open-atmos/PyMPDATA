from PyMPDATA_examples.Jarecka_et_al_2015 import Settings, Simulation

def test_just_do_it():
    settings = Settings()
    simulation = Simulation(settings)
    simulation.run()

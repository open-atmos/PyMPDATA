from joblib import Parallel, delayed
from MPyDATA_examples.Molenkamp_test_as_in_Jaruga_et_al_2015_Fig_12.simulation import Simulation
from MPyDATA_examples.Molenkamp_test_as_in_Jaruga_et_al_2015_Fig_12.setup import Setup
from MPyDATA.options import Options

# TODO
options = {
    'a': Options(n_iters=-1),
    'b': Options(n_iters=2, flux_corrected_transport=True),
    'c': Options(n_iters=3, flux_corrected_transport=True),  # TODO: tot=True
    'd': Options(n_iters=2, flux_corrected_transport=True, infinite_gauge=True)
}


def compute_panel(panel):
    setup = Setup(n_rotations=6)
    simulation = Simulation(setup, options[panel])
    if panel == 'a':
        return simulation.state
    simulation.run()
    return simulation.state


def fig_12_data():
    data = Parallel(n_jobs=-2)(
        delayed(compute_panel)(panel)
        for panel in ['a', 'b', 'c', 'd']
    )
    return data

from PyMPDATA_examples.Molenkamp_test_as_in_Jaruga_et_al_2015_Fig_12.simulation import Simulation
from PyMPDATA_examples.Molenkamp_test_as_in_Jaruga_et_al_2015_Fig_12.settings import Settings
from PyMPDATA import Options
from joblib import Parallel, delayed


options = {
    'upwind': Options(n_iters=1),
    '2+fct': Options(n_iters=2, flux_corrected_transport=True),
    '3+fct+tot': Options(n_iters=3, flux_corrected_transport=True, third_order_terms=True),
    '2+fct+iga': Options(n_iters=2, flux_corrected_transport=True, infinite_gauge=True)
}


def compute_panel(panel):
    settings = Settings(n_rotations=6)
    simulation = Simulation(settings, options[panel])
    if panel == 'upwind':
        return simulation.state
    simulation.run()
    return simulation.state


def fig_12_data():
    data = Parallel(n_jobs=-2)(
        delayed(compute_panel)(panel)
        for panel in ['upwind', '2+fct', '3+fct+tot', '2+fct+iga']
    )
    return data

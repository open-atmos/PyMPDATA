import os
os.environ['NUMBA_DEBUGINFO'] = '1'

from joblib import Parallel, delayed
from MPyDATA_examples.Molenkamp_test_as_in_Jaruga_et_al_2015_Fig_12.simulation import Simulation
from MPyDATA_examples.Molenkamp_test_as_in_Jaruga_et_al_2015_Fig_12.setup import Setup
from MPyDATA.options import Options

opts = {
    'a': {'options': Options(), 'n_iters': -1},
    'b': {'options': Options(fct=True), 'n_iters': 2},
    'c': {'options': Options(fct=True), 'n_iters': 3},  # TODO: tot=True
    'd': {'options': Options(fct=True, iga=True), 'n_iters': 2}
}


def compute_panel(panel):
    setup = Setup(n_rotations=6)
    simulation = Simulation(setup, **opts[panel])
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


if __name__ == '__main__':

    compute_panel('b')
from numpy.core._multiarray_umath import ndarray

from MPyDATA_examples.Olesik_et_al_2020.setup import Setup, default_nr, default_GC_max
from MPyDATA_examples.Olesik_et_al_2020.coordinates import x_id, x_log_of_pn, x_p2
from MPyDATA_examples.Olesik_et_al_2020.simulation import Simulation
from MPyDATA_examples.Olesik_et_al_2020.analysis import analysis, Case
from MPyDATA import Options
from MPyDATA_examples.Olesik_et_al_2020.setup import Setup, default_mixing_ratios_g_kg
import numpy as np

grid_layout_set = (x_id(), x_p2(), x_log_of_pn(base=2))
opt_set = (
    {'n_iters': 1},
    {'n_iters':2},
    {'n_iters':2,'infinite_gauge':True},
    {'n_iters': 2, 'infinite_gauge': True, 'flux_corrected_transport': True},
    {'n_iters':2, 'third_order_terms':True},
    {'n_iters':3},
    {'n_iters': 3, 'third_order_terms': True, 'infinite_gauge': True, 'flux_corrected_transport': True}
)



def test_wall_time():
    setup = Setup(nr=default_nr, mixing_ratios_g_kg=default_mixing_ratios_g_kg)
    for grid in grid_layout_set:
        norm = [1, ]
        for opts in opt_set:
            result = make_data(setup,grid,opts)
            print("\nVariant:", opts, "\nGrid Layout:", grid, "\n")
            wall_times = result['wall_time']
            mean_time = np.nanmean(wall_times)
            min_time = np.nanmin(wall_times)
            if opts == {'n_iters': 1}:
                norm[0] = min_time
            print("mean time:", round(mean_time, 2), "\n")
            print("min time:", round(min_time, 2), "\n")
            print("elapsed real time (wrt upwind)", round(min_time/norm[0], 2), "\n")

def make_data(setup,grid,opts):
    options = Options(**opts)
    simulation = Simulation(setup=setup, grid_layout=grid, psi_coord=x_id(), opts=options, GC_max=default_GC_max)
    result = {"wall_time": []}
    last_step = 0
    for n_steps in simulation.out_steps:
        steps = n_steps - last_step
        wall_time = simulation.step(steps)
        last_step += steps
        result['wall_time'].append(wall_time)
    return result
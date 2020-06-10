from MPyDATA_examples.Olesik_et_al_2020.setup import Setup, default_nr, default_GC_max
from MPyDATA_examples.Olesik_et_al_2020.coordinates import x_id, x_log_of_pn, x_p2
from MPyDATA_examples.Olesik_et_al_2020.analysis import compute_figure_data
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
    result, setup = compute_figure_data(nr=default_nr, GC_max=default_GC_max, psi_coord=x_id(),
                                        grid_layouts=grid_layout_set,
                                        opt_set=opt_set)
    for coord in result.keys():
        for opt_i, opts in enumerate(result[coord]['numerical'].keys()):
            print("\nVariant:", opts, "\nGrid Layout:", coord)
            wall_times = result[coord]['wall_time'][opts]
            mean_time = np.nanmean(wall_times)
            print("mean time:", mean_time, "\n")
            print("min time:", np.nanmin(wall_times), "\n")

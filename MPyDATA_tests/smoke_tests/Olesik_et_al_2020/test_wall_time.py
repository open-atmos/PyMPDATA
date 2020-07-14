from MPyDATA_examples.Olesik_et_al_2020.setup import Setup, default_nr, default_GC_max
from MPyDATA_examples.Olesik_et_al_2020.coordinates import x_id, x_log_of_pn
from MPyDATA_examples.Olesik_et_al_2020.simulation import Simulation
from MPyDATA import Options
from MPyDATA_examples.Olesik_et_al_2020.setup import Setup
import numpy as np
import pathlib
import pytest
import platform

grid_layout_set = (x_log_of_pn(base=2),)
opt_set = (
    {'n_iters': 1},
    {'n_iters': 2},
    {'n_iters': 2, 'infinite_gauge': True},
    {'n_iters': 2, 'infinite_gauge': True, 'flux_corrected_transport': True},
    {'n_iters': 2, 'third_order_terms': True},
    {'n_iters': 3},
    {'n_iters': 3, 'third_order_terms': True, 'infinite_gauge': True, 'flux_corrected_transport': True}
)


@pytest.mark.skipif(platform.system() == 'Windows',
                    reason="not enough accuracy on windows code (look MPyDATA/clock.py)")
def test_wall_time(n_runs=5, mrats=[1.5, ], generate=False, print_tab=False):
    setup = Setup(nr=default_nr, mixing_ratios_g_kg=np.array(mrats))
    table_data = {"opts": [], "values": []}
    for grid in grid_layout_set:
        norm = [1, ]
        for opts in opt_set:
            i = 0
            minimum_values = []
            while i < n_runs:
                result = make_data(setup, grid, opts)
                wall_times = result['wall_time']
                minimal = np.nanmin(wall_times)
                minimum_values.append(minimal)
                i += 1
            print(minimum_values)
            selected_value = np.min(minimum_values)
            if opts == {'n_iters': 1}:
                norm[0] = selected_value
            table_data["opts"].append(str(opts) + "(" + grid.__class__.__name__ + ")")
            table_data["values"].append(round(selected_value / norm[0], 2))
            print(table_data["values"])
    make_refdata(data=table_data, generate=generate, print_tab=print_tab)
    compare_refdata(data=table_data["values"])


def make_data(setup, grid, opts):
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


def make_refdata(data, generate=False, print_tab=False):
    latex_data = r"\hline" + " Variant  & Elapsed Real Time (wrt upwind) " + r"\\ \hline" + "\n"
    for opt, value in zip(data["opts"], data["values"]):
        latex_data += r"\hline" + f" {opt} & {value} " + r"\\ \hline" + "\n"
    latex_start = r"\begin" + "\n" + "{table}[]" + "\n" + r"\begin" + "\n" + "{tabular}" + "\n" + "{| l | l |}" + "\n"
    latex_end = "\end \n {tabular} \n \end \n {table}"
    latex_table = latex_start + latex_data + latex_end
    if print_tab:
        print(latex_table)
    with open(pathlib.Path(__file__).parent.joinpath("wall_time_refdata.txt"), "w+" if generate else "r") as f:
        if generate:
            f.write(latex_table)


def compare_refdata(data):
    decimal = .69  # abs(desired-actual) < 1.5 * 10**(-decimal) , so it would allow ~30% difference
    data_from_mybinder_test = [1.0, 2.53, 2.24, 5.76, 2.97, 4.09, 9.08]
    np.testing.assert_array_almost_equal(data_from_mybinder_test, data, decimal=decimal)

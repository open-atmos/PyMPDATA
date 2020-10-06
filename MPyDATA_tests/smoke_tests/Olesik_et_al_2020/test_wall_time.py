from MPyDATA_examples.Olesik_et_al_2020.setup import Setup, default_nr, default_GC_max, default_opt_set
from MPyDATA_examples.Olesik_et_al_2020.coordinates import x_id, x_log_of_pn
from MPyDATA_examples.Olesik_et_al_2020.simulation import Simulation
from MPyDATA import Options
import numpy as np
import pathlib
import platform, os


grid_layout_set = (x_log_of_pn(r0=1,base=2),)
opt_set = default_opt_set.values()

rtol = .25
if platform.system() != 'Linux' and 'TRAVIS' in os.environ:
    rtol = .75


def test_wall_time(n_runs=3, mrats=[10, ], generate=False, print_tab=True, rtol=rtol):
    setup = Setup(nr=default_nr * 10, mixing_ratios_g_kg=np.array(mrats))
    table_data = {"opts": [], "values": []}
    for grid in grid_layout_set:
        for opts in opt_set:
            i = 0
            minimum_values = []
            while i < n_runs:
                result = make_data(setup, grid, opts)
                wall_times = result['wall_time']
                minimal = np.nanmin(wall_times)
                minimum_values.append(minimal)
                i += 1
            selected_value = np.min(minimum_values)
            if opts == {'n_iters': 1}:
                norm = selected_value
            table_data["opts"].append(str(opts) + "(" + grid.__class__.__name__ + ")")
            table_data["values"].append(round(selected_value / norm, 1))
    make_textable(data=table_data, generate=generate, print_tab=print_tab)
    compare_refdata(data=table_data, rtol=rtol, generate=generate)


def make_data(setup, grid, opts):
    options = Options(**opts)
    simulation = Simulation(setup=setup, grid_layout=grid, psi_coord=x_id(), opts=options, GC_max=default_GC_max)
    result = {"wall_time": []}
    last_step = 0
    for n_steps in simulation.out_steps:
        steps = n_steps - last_step
        wall_time_per_timestep = simulation.step(steps)
        last_step += steps
        result['wall_time'].append(wall_time_per_timestep)
    return result


def make_textable(data, generate=False, print_tab=False):
    latex_data = r"\hline" + " Variant  & Elapsed Real Time (wrt upwind) " + r"\\ \hline" + "\n"
    for opt, value in zip(data["opts"], data["values"]):
        latex_data += r"\hline" + f" {opt} & {value} " + r"\\ \hline" + "\n"
    latex_start = r"\begin{table}[]" + "\n" + r"\begin{tabular}{| l | l |}" + "\n"
    latex_end = "\end{tabular} \n \end{table}"
    latex_table = latex_start + latex_data + latex_end
    if print_tab:
        print(latex_table)
    with open(pathlib.Path(__file__).parent.joinpath("wall_time_textable.txt"), "w+" if generate else "r") as f:
        if generate:
            f.write(latex_table)


def compare_refdata(data, rtol, generate=False):
    delimiter=';'
    path = pathlib.Path(__file__).parent.joinpath("wall_time_refdata.txt")
    if generate:
        table = np.char.array(np.concatenate([data['opts'], data['values']])).reshape(len(data['values']),2).T
        np.savetxt(path, table, delimiter=delimiter, fmt="%s")
    else:
        table = np.loadtxt(path, delimiter=delimiter, dtype=str)
        np.testing.assert_allclose(actual=data['values'], desired=np.array(table[:,1].astype(float)), rtol=rtol)
        np.testing.assert_array_equal(data['opts'], table[:,0])

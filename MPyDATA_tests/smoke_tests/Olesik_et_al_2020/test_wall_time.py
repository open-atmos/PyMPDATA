from MPyDATA_examples.Olesik_et_al_2020.setup import Setup, default_nr, default_GC_max, default_opt_set
from MPyDATA_examples.Olesik_et_al_2020.coordinates import x_id, x_log_of_pn
from MPyDATA_examples.Olesik_et_al_2020.simulation import Simulation
from MPyDATA import Options
import numpy as np
import pathlib


grid_layout_set = (x_log_of_pn(r0=1,base=2),)
opt_set = default_opt_set


def test_wall_time(n_runs=5, mrats=[5, ], generate=False, print_tab=True, rtol=.4):
    setup = Setup(nr=default_nr * 5, mixing_ratios_g_kg=np.array(mrats))
    print("nt=", setup.out_times[0])
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

            print("for travis testing upwind time", norm)
            table_data["values"].append(round(selected_value / norm, 1))
    make_textable(data=table_data, generate=generate, print_tab=print_tab)
    compare_refdata(data=table_data["values"], rtol=rtol, generate=generate)


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
    path = pathlib.Path(__file__).parent.joinpath("wall_time_refdata.txt")
    if generate:
        np.savetxt(path, data, delimiter=',')
    else:
        refdata = np.loadtxt(path, delimiter=',')[:len(data)]
        np.testing.assert_allclose(actual=data, desired=refdata, rtol=rtol)

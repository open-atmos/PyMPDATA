import pathlib

import numpy as np
from PyMPDATA_examples.Olesik_et_al_2022.coordinates import x_id, x_log_of_pn
from PyMPDATA_examples.Olesik_et_al_2022.settings import (
    Settings,
    default_GC_max,
    default_nr,
    default_opt_set,
)
from PyMPDATA_examples.Olesik_et_al_2022.simulation import Simulation

from PyMPDATA import Options

grid_layout_set = (x_log_of_pn(r0=1, base=2),)
opt_set = default_opt_set.values()

RTOL = 1.5


def test_wall_time(n_runs=3, mrats=(20,), generate=False, print_tab=True, rtol=RTOL):
    settings = Settings(nr=default_nr * 10, mixing_ratios_g_kg=np.array(mrats))
    table_data = {"opts": [], "values": []}
    for grid in grid_layout_set:
        for opts in opt_set:
            i = 0
            minimum_values = []
            while i < n_runs:
                result = make_data(settings, grid, opts)
                print("\t", i, opts, result['wall_time'])
                wall_times = result["wall_time"]
                minimal = np.nanmin(wall_times)
                minimum_values.append(minimal)
                i += 1
            selected_value = np.min(minimum_values)
            if opts == {"n_iters": 1}:
                norm = selected_value
                print(norm)
            table_data["opts"].append(str(opts) + "(" + grid.__class__.__name__ + ")")
            table_data["values"].append(
                np.nan if norm == 0 else round(selected_value / norm, 1)
            )
    make_textable(data=table_data, generate=generate, print_tab=print_tab)
    compare_refdata(data=table_data, rtol=rtol, generate=generate)


def make_data(settings, grid, opts):
    options = Options(**opts)
    simulation = Simulation(
        settings=settings,
        grid_layout=grid,
        psi_coord=x_id(),
        opts=options,
        GC_max=default_GC_max,
    )
    result = {"wall_time": []}
    last_step = 0
    for n_steps in simulation.out_steps:
        steps = n_steps - last_step
        wall_time_per_timestep = simulation.step(steps)
        last_step += steps
        result["wall_time"].append(wall_time_per_timestep)
    return result


def make_textable(data, generate=False, print_tab=False):
    latex_data = (
        r"\hline" + " Variant  & Elapsed Real Time (wrt upwind) " + r"\\ \hline" + "\n"
    )
    for opt, value in zip(data["opts"], data["values"]):
        latex_data += r"\hline" + f" {opt} & {value} " + r"\\ \hline" + "\n"
    latex_start = r"\begin{table}[]" + "\n" + r"\begin{tabular}{| l | l |}" + "\n"
    latex_end = r"\end{tabular}" + "\n" + r"\end{table}"
    latex_table = latex_start + latex_data + latex_end
    if print_tab:
        print(latex_table)
    with open(
        pathlib.Path(__file__).parent.joinpath("wall_time_textable.txt"),
        "w+" if generate else "r",
        encoding="utf-8",
    ) as f:
        if generate:
            f.write(latex_table)


def compare_refdata(data, rtol, generate=False):
    delimiter = ";"
    path = pathlib.Path(__file__).parent.joinpath("wall_time_refdata.txt")
    if generate:
        table = (
            np.char.array(np.concatenate([data["opts"], data["values"]]))
            .reshape(2, len(data["values"]))
            .T
        )
        np.savetxt(path, table, delimiter=delimiter, fmt="%s")
    else:
        table = np.loadtxt(path, delimiter=delimiter, dtype=str)
        np.testing.assert_allclose(
            actual=data["values"],
            desired=np.array(table[:, 1].astype(float)),
            rtol=rtol,
        )
        np.testing.assert_array_equal(data["opts"], table[:, 0])

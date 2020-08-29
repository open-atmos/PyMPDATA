from MPyDATA_examples.Olesik_et_al_2020.coordinates import x_id, x_log_of_pn, x_p2
from MPyDATA_examples.Olesik_et_al_2020.setup import Setup, default_mixing_ratios_g_kg
from MPyDATA import Options
from MPyDATA_examples.Olesik_et_al_2020.simulation import Simulation
from joblib import Parallel, parallel_backend, delayed
from MPyDATA_examples.Olesik_et_al_2020.physics.equilibrium_drop_growth import PdfEvolver
from MPyDATA_examples.utils.error_norms import L2
from MPyDATA.arakawa_c.discretisation import discretised_analytical_solution
from copy import deepcopy
import numpy as np


def analysis(setup, grid_layout, psi_coord, options_dict, GC_max):
    options_str = str(options_dict)
    options = Options(**options_dict)
    simulation = Simulation(setup, grid_layout,  psi_coord, options, GC_max)
    result = {"n": [], "n_analytical": [], "error_norm_L2": [], "wall_time":[]}
    last_step = 0
    for n_steps in simulation.out_steps:
        steps = n_steps - last_step
        wall_time = simulation.step(steps) if steps > 0 else 0
        last_step += steps
        result['n'].append(simulation.n_of_r.copy())
        result['wall_time'].append(wall_time)
    result['r'] = simulation.r.copy()
    result['rh'] = simulation.rh.copy()
    result['dx'] = simulation.dx
    return Result(
        grid_layout_str=grid_layout.__class__.__name__,
        option_str=options_str,
        result=result,
        out_steps=simulation.out_steps,
        dt=simulation.dt
    )


def compute_figure_data(*, nr, GC_max, psi_coord=x_id(),
                        grid_layouts=(x_id(), x_p2(), x_log_of_pn(r0=1, n=1)),
                        opt_set=({'n_iters': 1},),
                        mixing_ratios_g_kg=default_mixing_ratios_g_kg
                        ):
    setup = Setup(nr=nr, mixing_ratios_g_kg=mixing_ratios_g_kg)
    with parallel_backend('threading', n_jobs=-2):
        results = Parallel(verbose=10)(
            delayed(analysis)(setup, grid_layout, psi_coord, options, GC_max)
            for grid_layout in grid_layouts
            for options in deepcopy(opt_set)
        )

    cases = {}
    for result in results:
        case = Case(result)
        if case.grid_layour_str not in cases:
            cases[case.grid_layour_str] = case


    output = {}
    for coord, case in cases.items():
        output[coord] = {"numerical": {}, "wall_time": {}}
        for result in results:
            if coord == result.grid_layout_str:
                opts = result.option_str
                data = result.result
                rh = data.pop("rh")
                r = data.pop('r')
                dx = data.pop('dx')

                if 'grid' not in output[coord]:
                    output[coord]["grid"] = {'rh': rh, 'r': r, 'dx': dx, 'dt': case.dt, 'out_steps': case.out_steps}
                output[coord]["numerical"][opts] = data['n']

    for coord, case in cases.items():
        analytical = []
        for t in [case.dt * nt for nt in case.out_steps]:
            pdf_t = PdfEvolver(setup.pdf, setup.drdt, t)
            rh = output[coord]["grid"]['rh']
            analytical.append(discretised_analytical_solution(
                rh.magnitude,
                lambda r: pdf_t(r * rh.units).magnitude,
                midpoint_value=True,
                r=output[coord]["grid"]['r'].magnitude
            ) * pdf_t(rh[0]).units)   # TODO ? * coord.x (r * rh.units)
        output[coord]["analytical"] = analytical

    for coord, case in cases.items():
        error_L2 = {}
        analytical = output[coord]["analytical"]
        for opts in output[coord]["numerical"]:
            numerical = output[coord]["numerical"][opts]
            error_L2[opts] = L2(numerical[-1].magnitude, analytical[-1].magnitude, case.out_steps[-1])
        output[coord]["error_L2"] = error_L2

    for coord, case in cases.items():
        for result in results:
            data = result.result
            for opts in output[coord]["numerical"]:
                output[coord]["wall_time"][opts] = data["wall_time"]

    return output, setup


class Result:
    def __init__(self, *, dt, out_steps, grid_layout_str, option_str, result):
        self.dt = dt
        self.option_str = option_str
        self.result = result
        self.out_steps = out_steps
        self.grid_layout_str = grid_layout_str


class Case:
    def __init__(self, result: Result):
        self.dt = result.dt
        self.out_steps = result.out_steps
        self.grid_layour_str = result.grid_layout_str


def rel_disp(r, psi, psi_coord):
    mom0 = 0
    mom1 = 0
    mom2 = 0
    for i in range(len(psi)):
        psi_i = psi[i]
        dp_i = psi_coord.moment_of_r_integral(psi_coord.x(r[i+1]), 0) - psi_coord.moment_of_r_integral(psi_coord.x(r[i]), 0)
        A_i = psi_coord.moment_of_r_integral(psi_coord.x(r[i+1]), 1) - psi_coord.moment_of_r_integral(psi_coord.x(r[i]), 1)
        B_i = psi_coord.moment_of_r_integral(psi_coord.x(r[i+1]), 2) - psi_coord.moment_of_r_integral(psi_coord.x(r[i]), 2)
        bin_mom0 = psi_i * dp_i
        bin_mom1 = psi_i * A_i
        bin_mom2 = psi_i * B_i
        mom0 += bin_mom0
        mom1 += bin_mom1
        mom2 += bin_mom2
    mu = mom1 / mom0
    std = np.sqrt(mom2/mom0 - mu**2)
    d = std/mu
    return d


def third_moment(r, psi, psi_coord, normalised=True):
    mom0 = 0
    mom3 = 0
    for i in range(len(psi)):
        dp_i = psi_coord.moment_of_r_integral(psi_coord.x(r[i+1]), 0) - psi_coord.moment_of_r_integral(psi_coord.x(r[i]), 0)
        integral_i = psi_coord.moment_of_r_integral(psi_coord.x(r[i+1]), 3) - psi_coord.moment_of_r_integral(psi_coord.x(r[i]), 3)
        bin_mom0 = psi[i] * dp_i
        bin_mom3 = psi[i] * integral_i
        mom0 += bin_mom0
        mom3 += bin_mom3
    if normalised:
        mom3 /= mom0
    return mom3

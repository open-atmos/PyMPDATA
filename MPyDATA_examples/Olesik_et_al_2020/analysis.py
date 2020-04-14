from MPyDATA_examples.Olesik_et_al_2020.coordinates import x_id, x_log_of_pn, x_p2
from MPyDATA_examples.Olesik_et_al_2020.setup import Setup, default_mixing_ratios_g_kg
from MPyDATA.options import Options
from MPyDATA_examples.Olesik_et_al_2020.simulation import Simulation
from joblib import Parallel, parallel_backend, delayed
from MPyDATA_examples.Olesik_et_al_2020.physics.equilibrium_drop_growth import PdfEvolver
from MPyDATA_examples.utils.error_norms import L2
from MPyDATA.arakawa_c.discretisation import discretised_analytical_solution
from copy import deepcopy


def analysis(setup, grid_layout, psi_coord, options_dict):
    options_str = str(options_dict)
    options = Options(**options_dict)
    simulation = Simulation(setup, grid_layout, psi_coord, options)
    result = {"n": [], "n_analytical": [], "error_norm_L2": []}
    last_step = 0
    for n_steps in setup.out_steps:
        steps = n_steps - last_step
        simulation.step(steps)
        last_step += steps
        result['n'].append(simulation.n.copy())
    result['r'] = simulation.r.copy()
    result['rh'] = simulation.rh.copy()
    result['dx'] = simulation.dx
    return grid_layout.__class__.__name__, options_str, result


def compute_figure_data(*, nr, dt, psi_coord=x_id(),
                        grid_layouts=(x_id(), x_p2(), x_log_of_pn(n=1)),
                        opt_set=({'n_iters': 1},),
                        mixing_ratios_g_kg = default_mixing_ratios_g_kg
                        ):
    setup = Setup(nr=nr, dt=dt, mixing_ratios_g_kg = mixing_ratios_g_kg)
    with parallel_backend('threading', n_jobs=-2):
        results = Parallel(verbose=10)(
            delayed(analysis)( setup, grid_layout, psi_coord, options)
            for grid_layout in grid_layouts
            for options in deepcopy(opt_set)
        )

    coords = []
    for result in results:
        coord = result[0]
        if coord not in coords:
            coords.append(coord)

    output = {}
    for coord in coords:
        output[coord] = {"numerical": {}}
        for result in results:
            if coord == result[0]:
                opts = result[1]
                data = result[2]
                rh = data.pop("rh")
                r = data.pop('r')
                dx = data.pop('dx')
                if 'grid' not in output[coord]:
                    output[coord]["grid"] = {'rh': rh, 'r': r, 'dx': dx, 'dt': dt}
                output[coord]["numerical"][opts] = data['n']

    for coord in coords:
        analytical = []
        for t in [setup.dt * nt for nt in setup.out_steps]:
            pdf_t = PdfEvolver(setup.pdf, setup.drdt, t)
            rh = output[coord]["grid"]['rh']
            analytical.append(discretised_analytical_solution(
                rh.magnitude,
                lambda r: pdf_t(r * rh.units).magnitude
            ) * pdf_t(rh[0]).units)   # TODO ? * coord.x (r * rh.units)
        output[coord]["analytical"] = analytical

    for coord in coords:
        error_L2 = {}
        analytical = output[coord]["analytical"]
        for opts in output[coord]["numerical"]:
            numerical = output[coord]["numerical"][opts]
            error_L2[opts] = L2(numerical[-1].magnitude, analytical[-1].magnitude, setup.out_steps[-1])
        output[coord]["error_L2"] = error_L2

        # # TODO: calculate norms for mass and number

    return output, setup
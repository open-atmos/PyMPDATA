from MPyDATA_examples.Olesik_et_al_2020.coordinates import x_id, x_log_of_pn, x_p2
from MPyDATA_examples.Olesik_et_al_2020.setup import Setup, default_mixing_ratios_g_kg
from MPyDATA.options import Options
from MPyDATA_examples.Olesik_et_al_2020.simulation import Simulation
from joblib import Parallel, parallel_backend, delayed
from MPyDATA_examples.Olesik_et_al_2020.physics.equilibrium_drop_growth import PdfEvolver
from MPyDATA_examples.utils.error_norms import L2
from MPyDATA.arakawa_c.discretisation import discretised_analytical_solution
from copy import deepcopy


def analysis(setup, grid_layout, psi_coord, options_dict, GC_max):
    options_str = str(options_dict)
    options = Options(**options_dict)
    simulation = Simulation(setup, grid_layout,  psi_coord, options, GC_max)
    result = {"n": [], "n_analytical": [], "error_norm_L2": []}
    last_step = 0
    for n_steps in simulation.out_steps:
        steps = n_steps - last_step
        simulation.step(steps)
        last_step += steps
        result['n'].append(simulation.n.copy())
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
                        grid_layouts=(x_id(), x_p2(), x_log_of_pn(n=1)),
                        opt_set=({'n_iters': 1},),
                        mixing_ratios_g_kg=default_mixing_ratios_g_kg
                        ):
    setup = Setup(nr=nr, mixing_ratios_g_kg=mixing_ratios_g_kg)
    with parallel_backend('threading', n_jobs=-2):    # TODO: possible error with parallelisation
        results = Parallel(verbose=10)(
            delayed(analysis)( setup, grid_layout, psi_coord, options, GC_max)
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
        output[coord] = {"numerical": {}}
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
                lambda r: pdf_t(r * rh.units).magnitude
            ) * pdf_t(rh[0]).units)   # TODO ? * coord.x (r * rh.units)
        output[coord]["analytical"] = analytical

    for coord, case in cases.items():
        error_L2 = {}
        analytical = output[coord]["analytical"]
        for opts in output[coord]["numerical"]:
            numerical = output[coord]["numerical"][opts]
            error_L2[opts] = L2(numerical[-1].magnitude, analytical[-1].magnitude, case.out_steps[-1])
        output[coord]["error_L2"] = error_L2

        # # TODO: calculate norms for mass and number

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
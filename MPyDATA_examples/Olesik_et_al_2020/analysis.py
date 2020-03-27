from MPyDATA_examples.Olesik_et_al_2020.coordinates import x_id, x_ln, x_p2
from MPyDATA_examples.Olesik_et_al_2020.setup import Setup, default_nr, default_dt
from MPyDATA.options import Options
from MPyDATA_examples.Olesik_et_al_2020.simulation import Simulation
from joblib import Parallel, parallel_backend, delayed
from MPyDATA_examples.Olesik_et_al_2020.physics.equilibrium_drop_growth import PdfEvolver
from MPyDATA_examples.utils.error_norms import L2
from MPyDATA.utils.pdf_integrator import discretised_analytical_solution


def analysis(debug, setup, grid_layout, psi_coord, options_dict):
    options_str = str(options_dict)
    n_iters = options_dict.pop("n_iters")
    options = Options(nug=True, **options_dict)
    simulation = Simulation(setup, grid_layout, psi_coord, options)
    result = {"n": [], "n_analytical": [], "error_norm_L2": []}
    last_step = 0
    for n_steps in setup.nt:
        steps = n_steps - last_step
        for _ in range(steps):
            simulation.step(n_iters=n_iters, debug=debug)
        last_step += steps
        result['n'].append(simulation.n.copy())
    result['r'] = simulation.r.copy()
    result['rh'] = simulation.rh.copy()
    result['dx'] = simulation.dx
    return grid_layout.__class__.__name__, options_str, result

opt_set =  ({'n_iters': 1},{'n_iters':2},{'n_iters':3})
def compute_figure_data(*, debug=False, nr=default_nr, dt=default_dt, psi_coord=x_id(), opt_set=opt_set):
    setup = Setup(nr=nr, dt=dt)
    opt_set = opt_set
    with parallel_backend('threading'):
        results = Parallel(n_jobs=-2)(
            delayed(analysis)(debug, setup, grid_layout, psi_coord, options)
            for grid_layout in [x_id(), x_p2(), x_ln()]
            for options in  opt_set

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
                    output[coord]["grid"] = {'rh': rh, 'r': r, 'dx': dx}
                output[coord]["numerical"][opts] = data['n']

    for coord in coords:
        analytical = []
        for t in [setup.dt * nt for nt in setup.nt]:
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
            error_L2[opts] = L2(numerical[-1].magnitude, analytical[-1].magnitude, setup.nt[-1], setup.nr)
        output[coord]["error_L2"] = error_L2

        # # TODO: calculate norms for mass and number

    return output, setup


from MPyDATA_examples.condensational_growth.coord import x_id, x_ln, x_p2
from MPyDATA_examples.condensational_growth.setup import Setup, default_nr, default_dt
from MPyDATA.options import Options
from MPyDATA_examples.condensational_growth.simulation import Simulation
from joblib import Parallel, parallel_backend, delayed
from MPyDATA_examples.condensational_growth.physics.equilibrium_drop_growth import PdfEvolver
from MPyDATA_examples.utils.error_norms import L2


# <TODO> move
from scipy import integrate
import numpy as np


def discretised_analytical_solution(rh, pdf_t):
    output = np.empty(rh.shape[0]-1)
    for i in range(output.shape[0]):
        dcdf, _ = integrate.quad(pdf_t, rh[i], rh[i+1]) # TODO: handle other output values
        output[i] = dcdf / (rh[i+1] - rh[i])
    return output
# </TODO>


def analysis(debug, setup, coord, options_dict):
    options_str = str(options_dict)
    n_iters = options_dict.pop("n_iters")
    options = Options(nug=True, **options_dict)
    simulation = Simulation(setup, coord, options)
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
    return coord.__class__.__name__, options_str, result


def figure_data(debug=False, nr=default_nr, dt=default_dt):
    setup = Setup(nr=nr, dt=dt)
    with parallel_backend('threading'):
        results = Parallel(n_jobs=-2)(
            delayed(analysis)(debug, setup, coord, options)
            for coord in [x_id(), x_p2(), x_ln()]
            for options in (
                {'n_iters': 1},
                # {'n_iters': 2, 'fct': True},
                # {'n_iters': 3, 'dfl': True},
                # {'n_iters': 2, 'tot': True, 'iga': True, 'fct': True}
            )
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
                if 'grid' not in output[coord]:
                    output[coord]["grid"] = {'rh': rh, 'r': r}
                output[coord]["numerical"][opts] = data['n']

    for coord in coords:
        analytical = []
        for t in [setup.dt * nt for nt in setup.nt]:
            pdf_t = PdfEvolver(setup.pdf, setup.drdt, t)
            rh = output[coord]["grid"]['rh']
            analytical.append(discretised_analytical_solution(
                rh.magnitude,
                lambda r: pdf_t(r * rh.units).magnitude
            ) * pdf_t(rh[0]).units)
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


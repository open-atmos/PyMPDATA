from MPyDATA_examples.condensational_growth.coord import x_id, x_ln, x_p2
from MPyDATA_examples.condensational_growth.setup import setup
from MPyDATA.options import Options
from MPyDATA_examples.condensational_growth.simulation import Simulation
from joblib import Parallel, parallel_backend, delayed


def analysis(debug, coord, options_dict):
    options_str = str(options_dict)
    n_iters = options_dict.pop("n_iters")
    options = Options(nug=True, **options_dict)
    simulation = Simulation(coord, options)
    result = {"n": []}
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


def figure_data(debug=False):
    with parallel_backend('threading'):
        results = Parallel(n_jobs=-2)(
            delayed(analysis)(debug, coord, options)
            for coord in [x_id(), x_p2(), x_ln()]
            for options in (
                {'n_iters': 1},
                {'n_iters': 2, 'fct': True},
                {'n_iters': 3, 'dfl': True},
                {'n_iters': 2, 'tot': True, 'iga': True, 'fct': True}
            )
        )

    coords = []
    for result in results:
        coord = result[0]
        if coord not in coords:
            coords.append(coord)

    output = {}
    for coord in coords:
        output[coord] = {}
        for result in results:
            if coord == result[0]:
                output[coord][result[1]] = result[2]

    return output


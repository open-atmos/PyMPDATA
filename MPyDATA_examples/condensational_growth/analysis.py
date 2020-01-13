from MPyDATA_examples.condensational_growth.coord import x_id, x_ln, x_p2
from MPyDATA_examples.condensational_growth.setup import Setup
from MPyDATA.options import Options
from MPyDATA_examples.condensational_growth.simulation import Simulation
import pint
# from joblib import Parallel, delayed


def analysis():
    si = pint.UnitRegistry()
    setup = Setup(si)
    options = Options(nug=True) # , dfl=True, #iga=True, fct=True, tot=True)
    simulation = Simulation(x_id(), setup, options)

    result = {'r': None, "n": []}
    last_step = 0
    for n_steps in setup.nt:
        for _ in range(n_steps - last_step):
            simulation.step(n_iters=2)  # TODO
        last_step += n_steps
        result['n'].append(simulation.n.copy())
    result['r'] = simulation.r.copy()
    result['rh'] = simulation.rh.copy()
    return result, setup


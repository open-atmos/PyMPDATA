from MPyDATA_examples.Arabas_and_Farhat_2020.simulation import Simulation
from MPyDATA_examples.Arabas_and_Farhat_2020.setup2_american_put import Setup
from MPyDATA_examples.Arabas_and_Farhat_2020.analysis_figures_2_and_3 import error_L2_norm
import numpy as np
from joblib import Parallel, delayed


def compute_row(T, S0):
    row = [T, S0]
    for C_opt in (.02, .01, .005):
        setup = Setup(T=T, C_opt=C_opt, S0=S0)
        simulation = Simulation(setup)
        f = simulation.run(n_iters=2)
        row.append(
            error_L2_norm(simulation.solvers, simulation.setup, simulation.S, simulation.nt, simulation.nx, n_iters=2))
        np.testing.assert_almost_equal(simulation.S[simulation.ix_match], S0)
    row.append(f[simulation.ix_match])
    row.append(setup.analytical_solution(S0))
    row.append(setup.analytical_solution(S0, amer=False))
    return row


def table_1_data():
    result = Parallel(n_jobs=-2)(
        delayed(compute_row)(T, S0)
        for T in (.25, .5, 3)
        for S0 in (80, 90, 100, 110, 120)
    )
    return result


from PyMPDATA_examples.Arabas_and_Farhat_2020.simulation import Simulation
from PyMPDATA_examples.Arabas_and_Farhat_2020.setup2_american_put import Settings
from PyMPDATA_examples.Arabas_and_Farhat_2020.analysis_figures_2_and_3 import error_L2_norm
import numpy as np
from joblib import Parallel, delayed, parallel_backend


def compute_row(simulations):
    S0 = simulations[0].settings.S0
    T = simulations[0].settings.T
    for i in range(1, len(simulations)):
        assert simulations[i].settings.T == T
        assert simulations[i].settings.S0 == S0
    row = [T, S0]
    f = None
    for simulation in simulations:
        f = simulation.run(n_iters=2)
        row.append(
            error_L2_norm(simulation.solvers, simulation.settings, simulation.S, simulation.nt, n_iters=2))
        np.testing.assert_almost_equal(simulation.S[simulation.ix_match], S0)
    row.append(f[simulations[-1].ix_match])
    row.append(simulations[0].settings.analytical_solution(S0))
    row.append(simulations[0].settings.analytical_solution(S0, amer=False))
    return row


def table_1_data():
    with parallel_backend('threading', n_jobs=-2):
        result = Parallel(verbose=10)(
            delayed(compute_row)(tuple(
                Simulation(Settings(T=T, C_opt=C_opt, S0=S0))
                for C_opt in (.02, .01, .005)
            ))
            for T in (.25, .5, 3)
            for S0 in (80, 90, 100, 110, 120)
        )
        return result


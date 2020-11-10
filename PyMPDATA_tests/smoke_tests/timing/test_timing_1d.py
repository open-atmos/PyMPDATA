from PyMPDATA.options import Options
from PyMPDATA_examples.Smolarkiewicz_2006_Figs_3_4_10_11_12.settings import Settings
from PyMPDATA_examples.Smolarkiewicz_2006_Figs_3_4_10_11_12.simulation import Simulation
import pytest
import numpy as np


@pytest.mark.parametrize("options", [
    Options(n_iters=1),
    Options(n_iters=2),
    Options(n_iters=3),
    Options(n_iters=4),

    Options(n_iters=2, infinite_gauge=True),
    Options(n_iters=3, infinite_gauge=True),

    Options(n_iters=2, flux_corrected_transport=True),
    Options(n_iters=3, flux_corrected_transport=True),

    Options(n_iters=2, divergent_flow=True),
    Options(n_iters=3, divergent_flow=True),

    Options(n_iters=2, third_order_terms=True),
    Options(n_iters=3, third_order_terms=True)
])
def test_timing_1d(benchmark, options):
    simulation = Simulation(Settings("cosine"), options)
    psi0 = simulation.stepper.advectee.get().copy()

    def set_psi():
        simulation.stepper.advectee.get()[:] = psi0

    benchmark.pedantic(simulation.run, {}, setup=set_psi, warmup_rounds=1, rounds=3)

    print(np.amin(simulation.state), np.amax(simulation.state))
    if not options.infinite_gauge:
        assert np.amin(simulation.state) >= 0

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import pytest
import numpy as np
import numba
from PyMPDATA_examples.Smolarkiewicz_1984 import Simulation, Settings
from PyMPDATA import Options

from .concurrency_fixture import num_threads
assert hasattr(num_threads, '_pytestfixturefunction')


@pytest.mark.parametrize("options", [
    {'n_iters': 1},
    # TODO #96
    # {'n_iters': 2},
    # {'n_iters': 3, 'infinite_gauge': True},
    # {'n_iters': 2, 'infinite_gauge': True, 'nonoscillatory': True},
    # {'n_iters': 3, 'infinite_gauge': False, 'third_order_terms': True},
    # {'n_iters': 3, 'infinite_gauge': True, 'third_order_terms': True, 'nonoscillatory': True},
])
@pytest.mark.parametrize("dtype", (np.float64, ))
@pytest.mark.parametrize("static", (True, False))
def test_timing_3d(benchmark, options, dtype, static, num_threads):
    numba.set_num_threads(num_threads)

    settings = Settings(n=20, dt=1)
    simulation = Simulation(settings, Options(**options, dtype=dtype), static=static)

    def reset():
        simulation.solver.advectee.get()[:] = settings.advectee

    nt = 10
    benchmark.pedantic(simulation.run, (nt,), setup=reset, warmup_rounds=1, rounds=1)

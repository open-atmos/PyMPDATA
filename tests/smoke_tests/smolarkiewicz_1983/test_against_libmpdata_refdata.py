# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import pytest
import numpy as np
from PyMPDATA_examples.Smolarkiewicz_1984 import Settings, Simulation
from PyMPDATA import Options


# https://github.com/igfuw/libmpdataxx/blob/master/tests/paper_2015_GMD/4_revolving_sphere_3d/...
STATS = {
    # ...refdata/stats_upwind.txt.gz
    Options(n_iters=1): {
        0: {
            'min(solution)': 0.00000000,
            'max(solution)': 4.00000000,
        },
        566: {
            'max(solution)': 1.72131033,
            'min(solution)': 0.00000000,
            'Linf': 3.38441916,
            'L2': 0.00567238,
            'L1': 0.00128141,
        }
    },
    # ...refdata/stats_basic.txt.gz
    Options(n_iters=2): {
        0: {
            'min(solution)': 0.00000000,
            'max(solution)': 4.00000000,
        },
        556: {
            'max(solution)': 4.94170863,
            'min(solution)': 0.00000000,
            'Linf': 2.92022414,
            'L2': 0.00367407,
            'L1': 0.00065310,
        }
    },
    # ...refdata/stats_fct.txt.gz
    Options(n_iters=2, nonoscillatory=True): {
        0: {
            'min(solution)': 0.00000000,
            'max(solution)': 4.00000000,
        },
        556: {
            'max(solution)': 3.99999989,
            'min(solution)': 0.00000000,
            'Linf': 2.90357355,
            'L2': 0.00365567,
            'L1': 0.00064920
        }
    },
    # ...refdata/stats_iga.txt.gz
    Options(n_iters=2, infinite_gauge=True): {
        0: {
            'min(solution)': 0.00000000,
            'max(solution)': 4.00000000,
        },
        556: {
            'max(solution)': 6.16075462,
            'min(solution)': -1.01495101,
            'Linf': 2.94529169,
            'L2': 0.00328204,
            'L1': 0.00064378
        }
    },
    # ...refdata/stats_iga_fct.txt.gz
    Options(n_iters=2, infinite_gauge=True, nonoscillatory=True): {
        0: {
            'min(solution)': 0.00000000,
            'max(solution)': 4.00000000
        },
        556: {
            'max(solution)': 3.99999978,
            'min(solution)': 0.00000000,
            'Linf': 2.74523808,
            'L2': 0.00281070,
            'L1': 0.00038398
        }
    }
}

SETTINGS = Settings(n=59+1, dt=0.018 * 2 * np.pi)


@pytest.mark.parametrize('options', tuple(pytest.param(opt, id=str(opt)) for opt in STATS))
def test_against_libmpdata_refdata(options):
    # arrange
    simulation = Simulation(SETTINGS, options)
    actual = {}

    # act
    steps_done = 0
    for timesteps in STATS[options]:
        simulation.run(nt=timesteps - steps_done)
        steps_done += timesteps
        psi = simulation.solver.advectee.get()
        absdiff = np.abs(psi - SETTINGS.advectee)
        volume = np.product(SETTINGS.grid)
        time = steps_done * SETTINGS.dt
        actual[steps_done] = {
            'min(solution)': np.amin(psi),
            'max(solution)': np.amax(psi)
        }
        if steps_done > 0:
            actual[steps_done]['Linf'] = np.amax(absdiff)
            actual[steps_done]['L1'] = 1 / time * (1 / volume * np.sum((absdiff)**1))
            actual[steps_done]['L2'] = 1 / time * (1 / volume * np.sum((absdiff)**2))**.5

    # assert
    for step in STATS[options].keys():
        for stat in STATS[options][step].keys():
            np.testing.assert_approx_equal(
                desired=STATS[options][step][stat],
                actual=actual[step][stat],
                significant=1,  # TODO #96
                err_msg=stat
            )

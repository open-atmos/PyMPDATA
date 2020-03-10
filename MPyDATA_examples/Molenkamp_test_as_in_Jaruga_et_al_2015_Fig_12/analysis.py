# import os
# os.environ['NUMBA_DEBUGINFO'] = '1'

from MPyDATA_examples.Molenkamp_test_as_in_Jaruga_et_al_2015_Fig_12.simulation import Simulation
from MPyDATA_examples.Molenkamp_test_as_in_Jaruga_et_al_2015_Fig_12.setup import Setup, h0, h
import numpy as np
import os
# os.environ['NUMBA_DISABLE_JIT'] = '1'


def compute_panel():
    setup = Setup(n_rotations=6)
    simulation = Simulation(setup)
    simulation.run()

    state = simulation.state

    print(np.amin(state), np.amax(state))

    assert np.amin(state) >= h0
    np.testing.assert_almost_equal(np.amax(state), 3.7111, decimal=4)

    if False:
        from matplotlib import pyplot
        pyplot.imshow(state)
        pyplot.colorbar()
        pyplot.show()

    return state


if __name__ == '__main__':
    compute_panel()

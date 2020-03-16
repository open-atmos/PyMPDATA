from MPyDATA.mpdata_factory import MPDATAFactory
#from MPyDATA.arakawa_c.boundary_conditions.cyclic import CyclicLeft, CyclicRight
#from MPyDATA.options import Options
import numpy as np
import pytest


class TestMPDATA1D:
    @pytest.mark.skip()
    def test_fct_init(self):
        # Arrange
        state = np.array([1, 2, 3])
        C = 0
        opts = Options(fct=True)
        sut = MPDATAFactory.uniform_C_1d(state, C, opts, boundary_conditions=((CyclicLeft(), CyclicRight()),))
        sut.arrays.prev.swap_memory(sut.arrays.curr)
        sut.arrays.prev.fill_halos()

        # Act
        sut.fct_init(sut.arrays.prev, n_iters=2)

        # Assert
        np.testing.assert_equal(np.array([3]*5), sut.arrays.psi_max._impl.data[1:-1])
        np.testing.assert_equal(np.array([1]*5), sut.arrays.psi_min._impl.data[1:-1])

    def test_TODO(self):
        state = np.array([0, 1, 0])
        C = 1

        mpdata = MPDATAFactory.constant_1d(state, C)
        nt = 5

        conserved = np.sum(mpdata.curr.get())
        mpdata.step(nt)

        assert np.sum(mpdata.curr.get()) == conserved

"""
Created at 11.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from MPyDATA.mpdata_factory import MPDATAFactory
from MPyDATA.opts import Opts

from MPyDATA.fields import scalar_field, vector_field
import numpy as np
import pytest

# noinspection PyUnresolvedReferences
from MPyDATA_tests.__parametrisation__ import halo


class TestMPDATA2D:
    @pytest.mark.parametrize("shape, ij0, out, C, n_steps", [
        pytest.param((3, 1), (1, 0), np.array([[0.], [0.], [44.]]), (1., 0.), 1),
        pytest.param((1, 3), (0, 1), np.array([[0., 0., 44.]]), (0., 1.), 1),
        pytest.param((1, 3), (0, 1), np.array([[44., 0., 0.]]), (0., -1.), 1),
        pytest.param((3, 3), (1, 1), np.array([[0, 0, 0], [0, 0, 22], [0., 22., 0.]]), (.5, .5), 1),
        pytest.param((3, 3), (1, 1), np.array([[0, 0, 0], [0, 0, 22], [0., 22., 0.]]), (.5, .5), 1),
    ])
    def test(self, shape, ij0, out, C, n_steps, halo):
        value = 44
        scalar_field_init = np.zeros(shape)
        scalar_field_init[ij0] = value

        vector_field_init_x = np.full((shape[0] + 1, shape[1]), C[0])
        vector_field_init_y = np.full((shape[0], shape[1] + 1), C[1])
        state = scalar_field.make(scalar_field_init, halo=halo)
        GC_field = vector_field.make((vector_field_init_x, vector_field_init_y), halo=halo)

        G = scalar_field.make(np.ones(shape), halo=0)
        mpdata = MPDATAFactory._mpdata(GC_field=GC_field, state=state, g_factor=G, opts=Opts(n_iters=1))
        mpdata.debug_print()
        for _ in range(n_steps):
            mpdata.step()
        mpdata.debug_print()

        np.testing.assert_array_equal(mpdata.curr.data[halo:shape[0]-halo+2, halo:shape[1]-halo+2], out)

"""
Created at 11.10.2019

@author: Piotr Bartman
@author: Michael Olesik
@author: Sylwester Arabas
"""

from MPyDATA.arakawa_c.scalar_field import ScalarField
from MPyDATA.arakawa_c.vector_field import VectorField
from MPyDATA.mpdata import MPDATA
from MPyDATA.mpdata_factory import MPDATAFactory

import numpy as np
import pytest

# noinspection PyUnresolvedReferences
from MPyDATA_tests.unit_tests.__parametrisation__ import halo, case


@pytest.fixture(scope="module")
def options():
    return Options()


class TestMPDATA2D:
    @pytest.mark.skip()
    @pytest.mark.parametrize("shape, ij0, out, C, n_steps", [
        pytest.param((3, 1), (1, 0), np.array([[0.], [0.], [44.]]), (1., 0.), 1),
        pytest.param((1, 3), (0, 1), np.array([[0., 0., 44.]]), (0., 1.), 1),
        pytest.param((1, 3), (0, 1), np.array([[44., 0., 0.]]), (0., -1.), 1),
        pytest.param((3, 3), (1, 1), np.array([[0, 0, 0], [0, 0, 22], [0., 22., 0.]]), (.5, .5), 1),
        pytest.param((3, 3), (1, 1), np.array([[0, 0, 0], [0, 0, 22], [0., 22., 0.]]), (.5, .5), 1),
    ])
    def test_44(self, shape, ij0, out, C, n_steps, halo, options):
        value = 44
        scalar_field_init = np.zeros(shape)
        scalar_field_init[ij0] = value

        bcond = (
            (CyclicLeft(), CyclicRight()),
            (CyclicLeft(), CyclicRight()),
        )

        vector_field_init_x = np.full((shape[0] + 1, shape[1]), C[0])
        vector_field_init_y = np.full((shape[0], shape[1] + 1), C[1])
        state = ScalarField(scalar_field_init, halo=halo, boundary_conditions=bcond)
        GC_field = VectorField((vector_field_init_x, vector_field_init_y), halo=halo, boundary_conditions=bcond)

        G = ScalarField(np.ones(shape), halo=0, boundary_conditions=bcond)
        mpdata = MPDATA(advector=GC_field, advectee=state, g_factor=G, opts=options)
        for _ in range(n_steps):
            mpdata.step(n_iters=1)

        np.testing.assert_array_equal(
            mpdata.arrays.curr.get(),
            out
        )

    @pytest.mark.skip()
    def test_Arabas_et_al_2014_sanity(self, case, options):
        case = {
            "nx": case[0],
            "ny": case[1],
            "Cx": case[2],
            "Cy": case[3],
            "nt": case[4],
            "ni": case[5],
            "input": case[6],
            "output": case[7]
        }
        # Arrange
        sut = MPDATAFactory.uniform_C_2d(
            case["input"].reshape((case["nx"], case["ny"])),
            [case["Cx"], case["Cy"]],
            options
        )

        # Act
        for _ in range(case["nt"]):
            sut.step(n_iters=case["ni"])

        # Assert
        np.testing.assert_almost_equal(sut.arrays.curr.get(), case["output"].reshape(case["nx"], case["ny"]), decimal=4)

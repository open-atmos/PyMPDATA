from MPyDATA import ScalarField, VectorField, PolarBoundaryCondition, PeriodicBoundaryCondition
from MPyDATA.arakawa_c.traversals import Traversals
import numpy as np
import pytest

LEFT, RIGHT = 'left', 'right'


class TestPeriodicBoundaryCondition:
    @pytest.mark.parametrize("halo", (1, ))
    @pytest.mark.parametrize("side", (LEFT, RIGHT))
    def test_scalar_2d(self, halo, side):
        # arrange
        data = np.array(
            [
                [1, 5],
                [2, 6],
                [3, 7],
                [4, 8]
            ]
        )
        bc = (
            PeriodicBoundaryCondition(),
            PolarBoundaryCondition(data.shape, 0)
        )
        field = ScalarField(data, halo, bc)
        meta_and_data, fill_halos = field.impl
        traversals = Traversals(grid=data.shape, halo=halo, jit_flags={})
        sut, _ = traversals.make_boundary_conditions()

        # act
        sut(*meta_and_data, *fill_halos)

        # assert
        print(field.data)
        # TODO
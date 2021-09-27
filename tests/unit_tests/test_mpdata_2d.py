import pytest
import numpy as np
from PyMPDATA import VectorField, PeriodicBoundaryCondition, ScalarField, Stepper, Solver
from PyMPDATA.options import Options

# test data by Dorota Jarecka
# see http://dx.doi.org/10.3233/SPR-140379
params = (
    (3, 3, 0.1, 0.5, 1, 1, False,
     np.array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 1.]]),
     np.array([[0., 0., 0.1],
               [0., 0., 0.],
               [0.5, 0., 0.4]])
     ),
    (3, 3, 0.1, 0.5, 1, 2, False,
     np.array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 1.]]),
     np.array([[0., 0., 0.0921],
               [0., 0., 0.],
               [0.5011, 0., 0.4068]])
     ),
    (3, 3, 0.1, 0.5, 1, 2, True,
     np.array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 1.]]),
     np.array([[0., 0., 0.0946],
               [0., 0., 0.],
               [0.5111, 0., 0.3943]])
     ),
    (3, 3, 0.2, 0.2, 1, 1, False,
     np.array([[0., 0., 0.],
               [0., 1., 0.],
               [0., 1., 0.]]),
     np.array([[0., 0.2, 0.],
               [0., 0.6, 0.2],
               [0., 0.8, 0.2]])
     ),
    (3, 3, 0.2, 0.2, 1, 2, False,
     np.array([[0., 0., 0.],
               [0., 1., 0.],
               [0., 0., 0.]]),
     np.array([[0., 0., 0.],
               [0., 0.64, 0.18],
               [0., 0.18, 0.]])
     ),
    (3, 3, 0.2, 0.2, 1, 3, False,
     np.array([[0., 0., 0.],
               [0., 1., 0.],
               [0., 0., 0.]]),
     np.array([[0., 0., 0.],
               [0., 0.6578, 0.1711],
               [0., 0.1711, 0.]])
     ),
    (3, 3, 0.5, 0.5, 1, 1, False,
     np.array([[0., 0., 0.],
               [0., 1., 0.],
               [0., 0., 0.]]),
     np.array([[0., 0., 0.],
               [0., 0., 0.5],
               [0., 0.5, 0.]])),
    (3, 3, 0.5, 0.5, 1, 2, False,
     np.array([[0., 0., 0.],
               [0., 1., 0.],
               [0., 0., 0.]]),
     np.array([[0., 0., 0.],
               [0., 0., 0.5],
               [0., 0.5, 0.]])
     ),
    (3, 3, 0.5, 0.5, 1, 3, False, np.array([[0., 0., 0.],
                                     [0., 1., 0.],
                                     [0., 0., 0.]]), np.array([[0., 0., 0.],
                                                               [0., 0., 0.5],
                                                               [0., 0.5, 0.]])),
    (3, 3, 0.0, 1.0, 3, 1, False, np.array([[0., 0., 0.],
                                     [0., 1., 0.],
                                     [0., 0., 0.]]), np.array([[0., 0., 0.],
                                                               [0., 1., 0.],
                                                               [0., 0., 0.]])),
    (3, 3, 0.0, 1.0, 3, 2, False, np.array([[0., 0., 0.],
                                     [0., 1., 0.],
                                     [0., 0., 0.]]), np.array([[0., 0., 0.],
                                                               [0., 1., 0.],
                                                               [0., 0., 0.]])),
    (3, 3, 0.0, 1.0, 3, 3, False, np.array([[0., 0., 0.],
                                     [0., 1., 0.],
                                     [0., 0., 0.]]), np.array([[0., 0., 0.],
                                                               [0., 1., 0.],
                                                               [0., 0., 0.]])),
    (3, 3, 1.0, 0.0, 4, 1, False, np.array([[0., 0., 0.],
                                     [0., 0., 0.],
                                     [0., 1., 0.]]), np.array([[0., 1., 0.],
                                                               [0., 0., 0.],
                                                               [0., 0., 0.]])),
    (3, 3, 1.0, 0.0, 4, 2, False, np.array([[0., 0., 0.],
                                     [0., 0., 0.],
                                     [0., 1., 0.]]), np.array([[0., 1., 0.],
                                                               [0., 0., 0.],
                                                               [0., 0., 0.]])),
    (3, 3, 1.0, 0.0, 4, 3, False, np.array([[0., 0., 0.],
                                     [0., 0., 0.],
                                     [0., 1., 0.]]), np.array([[0., 1., 0.],
                                                               [0., 0., 0.],
                                                               [0., 0., 0.]])),
    (4, 4, 0.5, 0.5, 1, 1, False, np.array([[0., 0., 0., 0.],
                                     [0., 0., 0., 0.],
                                     [0., 0., 0., 0.],
                                     [0., 0., 0., 1.]]), np.array([[0., 0., 0., 0.5],
                                                                   [0., 0., 0., 0.],
                                                                   [0., 0., 0., 0.],
                                                                   [0.5, 0., 0., 0.]])),
    (4, 4, 0.5, 0.5, 1, 2, False, np.array([[0., 0., 0., 0.],
                                     [0., 0., 0., 0.],
                                     [0., 0., 0., 0.],
                                     [0., 0., 0., 1.]]), np.array([[0., 0., 0., 0.5],
                                                                   [0., 0., 0., 0.],
                                                                   [0., 0., 0., 0.],
                                                                   [0.5, 0., 0., 0.]])),
    (4, 4, 0.5, 0.5, 1, 3, False, np.array([[0., 0., 0., 0.],
                                     [0., 0., 0., 0.],
                                     [0., 0., 0., 0.],
                                     [0., 0., 0., 1.]]), np.array([[0., 0., 0., 0.5],
                                                                   [0., 0., 0., 0.],
                                                                   [0., 0., 0., 0.],
                                                                   [0.5, 0., 0., 0.]])),
    (10, 1, 1.0, 0.0, 5, 1, False, np.array([1., 0., 1., 0., 0., 0., 0., 0., 0., 0.]),
        np.array([0., 0., 0., 0., 0., 1., 0., 1., 0., 0.])))


@pytest.fixture(params=params)
def case_data(request):
    return request.param


# pylint: disable-next=redefined-outer-name
def test_Arabas_et_al_2014_sanity(case_data):
    case = {
        "nx": case_data[0],
        "ny": case_data[1],
        "Cx": case_data[2],
        "Cy": case_data[3],
        "nt": case_data[4],
        "ni": case_data[5],
        "dimsplit": case_data[6],
        "input": case_data[7],
        "output": case_data[8]
    }
    # Arrange
    data = case["input"].reshape((case["nx"], case["ny"]))
    c = [case["Cx"], case["Cy"]]
    options = Options(n_iters=case["ni"], dimensionally_split=case["dimsplit"])
    grid = data.shape
    advector_data = [
        np.full((grid[0] + 1, grid[1]), c[0], dtype=options.dtype),
        np.full((grid[0], grid[1] + 1), c[1], dtype=options.dtype)
    ]
    bcs = (PeriodicBoundaryCondition(), PeriodicBoundaryCondition())
    advector = VectorField(advector_data, halo=options.n_halo,
                           boundary_conditions=bcs)
    advectee = ScalarField(data=data.astype(dtype=options.dtype), halo=options.n_halo,
                           boundary_conditions=bcs)
    stepper = Stepper(options=options, grid=grid, non_unit_g_factor=False)
    mpdata = Solver(stepper=stepper, advectee=advectee, advector=advector)
    sut = mpdata

    # Act
    sut.advance(nt=case["nt"])

    # Assert
    np.testing.assert_almost_equal(
        sut.advectee.get(),
        case["output"].reshape(case["nx"], case["ny"]),
        decimal=4
    )

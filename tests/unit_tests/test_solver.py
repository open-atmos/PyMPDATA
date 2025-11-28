# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest

from PyMPDATA import Options, ScalarField, Solver, Stepper, VectorField
from PyMPDATA.boundary_conditions import Periodic

BCS = (Periodic(),)


@pytest.mark.parametrize(
    "case",
    (
        {"g_factor": None, "non_zero_mu_coeff": True, "mu": None},
        {"g_factor": None, "non_zero_mu_coeff": True, "mu": (0,)},
        pytest.param(
            {"g_factor": None, "non_zero_mu_coeff": False, "mu": (0,)},
            marks=pytest.mark.xfail(strict=True),
        ),
        pytest.param(
            {
                "g_factor": ScalarField(np.asarray([1.0, 1]), Options().n_halo, BCS),
                "non_zero_mu_coeff": True,
                "mu": None,
            },
            marks=pytest.mark.xfail(strict=True),
        ),
    ),
)
def test_mu_arg_handling(case):
    opt = Options(non_zero_mu_coeff=case["non_zero_mu_coeff"])
    advector = VectorField((np.asarray([1.0, 2, 3]),), opt.n_halo, BCS)
    advectee = ScalarField(np.asarray([4.0, 5]), opt.n_halo, BCS)
    stepper = Stepper(options=opt, n_dims=1)
    sut = Solver(stepper, advectee, advector, case["g_factor"])

    sut.advance(1, mu_coeff=case["mu"])


def test_multiple_scalar_fields():
    opt = Options()
    data = np.asarray([4.0, 5])
    advector = VectorField((np.asarray([1.0, 2, 3]),), opt.n_halo, BCS)
    advectees = [ScalarField(data, opt.n_halo, BCS)] * 5
    stepper = Stepper(options=opt, n_dims=1)
    sut = Solver(stepper, advectees, advector)

    sut.advance(1)

    for advectee in advectees:
        assert (advectee.get() != data).all()

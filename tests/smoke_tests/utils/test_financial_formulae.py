# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from PyMPDATA_examples.utils.financial_formulae import Black_Scholes_1973 as BS73


@pytest.mark.parametrize("funs", (
        {"normal": BS73.c_euro, "with_dividend":    BS73.c_euro_with_dividend},
        {"normal": BS73.p_euro, "with_dividend":    BS73.p_euro_with_dividend},
))
@pytest.mark.parametrize("S", (np.array([95, 100, 105]),))
@pytest.mark.parametrize("K", (100,10))
@pytest.mark.parametrize("T", (1,.5))
@pytest.mark.parametrize("r", (0.05,.001))
@pytest.mark.parametrize("sgma", (0.2,.5))
@pytest.mark.parametrize("dividend_yield", (0.02,0))
def test_black_scholes_with_dividend(funs: dict, S, K, T, r, sgma, dividend_yield):
    common_args = {"S": S, "K": K, "T": T, "sgma": sgma, "r": r}
    price_dividend = funs["with_dividend"](dividend_yield=dividend_yield, **common_args)
    price_normal = funs["normal"](b=r - dividend_yield, **common_args)
    assert np.allclose(price_dividend, price_normal)





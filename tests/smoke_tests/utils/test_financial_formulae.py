# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
import numpy as np
import pytest
from PyMPDATA_examples.utils.financial_formulae import Black_Scholes_1973 as BS73
from PyMPDATA_examples.utils.financial_formulae import asian_option as AO


class TestFinancialFormulae:
    @staticmethod
    @pytest.mark.parametrize(
        "funs",
        (
            {"normal": BS73.c_euro, "with_dividend": BS73.c_euro_with_dividend},
            {"normal": BS73.p_euro, "with_dividend": BS73.p_euro_with_dividend},
        ),
    )
    @pytest.mark.parametrize("S", (np.array([95, 100, 105]),))
    @pytest.mark.parametrize("K", (100, 10))
    @pytest.mark.parametrize("T", (1, 0.5))
    @pytest.mark.parametrize("r", (0.05, 0.001))
    @pytest.mark.parametrize("sgma", (0.2, 0.5))
    @pytest.mark.parametrize("dividend_yield", (0.02, 0))
    def test_black_scholes_with_dividend(funs: dict, S, K, T, r, sgma, dividend_yield):
        common_args = {"S": S, "K": K, "T": T, "sgma": sgma, "r": r}
        price_dividend = funs["with_dividend"](
            dividend_yield=dividend_yield, **common_args
        )
        price_normal = funs["normal"](b=r - dividend_yield, **common_args)
        assert np.allclose(price_dividend, price_normal)

    @staticmethod
    @pytest.mark.parametrize(
        "fun",
        (
            (AO.geometric_asian_average_price_c, 3.246),
            (AO.geometric_asian_average_price_p, 2.026),
            (AO.geometric_asian_average_strike_c, 3.725),
            (AO.geometric_asian_average_strike_p, 1.869),
        ),
    )
    @pytest.mark.parametrize("S", (40,))
    @pytest.mark.parametrize("K", (40,))
    @pytest.mark.parametrize("T", (1,))
    @pytest.mark.parametrize("r", (0.08,))
    @pytest.mark.parametrize("sgma", (0.3,))
    @pytest.mark.parametrize("dividend_yield", (0,))
    def test_asian_geometric_average(fun: callable, S, K, T, r, sgma, dividend_yield):
        """
        Analytic results are taken from [Derivatives Markets](
        https://faculty.ksu.edu.sa/sites/default/files/derivatives_markets_3e_0.pdf) page 413
        """
        price = fun[0](S=S, K=K, T=T, r=r, sgma=sgma, dividend_yield=dividend_yield)
        assert np.allclose(price, fun[1], rtol=1e-3)

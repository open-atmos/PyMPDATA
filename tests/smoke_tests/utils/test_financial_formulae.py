# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring
# pylint: disable=too-many-arguments,invalid-name
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
    @pytest.mark.parametrize("price", (np.array([95, 100, 105]),))
    @pytest.mark.parametrize("strike", (100, 10))
    @pytest.mark.parametrize("time_to_maturity", (1, 0.5))
    @pytest.mark.parametrize("risk_free_rate", (0.05, 0.001))
    @pytest.mark.parametrize("volatility", (0.2, 0.5))
    @pytest.mark.parametrize("dividend_yield", (0.02, 0))
    def test_black_scholes_with_dividend(
        funs: dict,
        price,
        strike,
        time_to_maturity,
        risk_free_rate,
        volatility,
        dividend_yield,
    ):
        common_args = {
            "S": price,
            "K": strike,
            "T": time_to_maturity,
            "sgma": volatility,
            "r": risk_free_rate,
        }
        price_dividend = funs["with_dividend"](
            dividend_yield=dividend_yield, **common_args
        )
        price_normal = funs["normal"](b=risk_free_rate - dividend_yield, **common_args)
        assert np.allclose(price_dividend, price_normal)

    @staticmethod
    @pytest.mark.parametrize(
        "fun, expected_value",
        (
            (AO.geometric_asian_average_price_c, 3.246),
            (AO.geometric_asian_average_price_p, 2.026),
            (AO.geometric_asian_average_strike_c, 3.725),
            (AO.geometric_asian_average_strike_p, 1.869),
        ),
    )
    def test_asian_geometric_average(fun: callable, expected_value):
        """
        Analytic results are taken from [Derivatives Markets](
        https://faculty.ksu.edu.sa/sites/default/files/derivatives_markets_3e_0.pdf) page 413
        """
        price = fun(S=40, K=40, T=1, r=0.08, sgma=0.3, dividend_yield=0)
        assert np.allclose(price, expected_value, rtol=1e-3)

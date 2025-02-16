# pylint: disable=line-too-long
"""
Closed-forms for geometric Asian options are taken from:
[McDonald 2013, "Derivatives Markets", Appendix 19A](https://media.pearsoncmg.com/ph/bp/bridgepages/teamsite/mcdonald/McDonald-web-19-A.pdf)
"""

import numpy as np

from .Black_Scholes_1973 import c_euro_with_dividend, p_euro_with_dividend


def geometric_asian_average_price_c(S, K, T, r, sgma, dividend_yield):
    return c_euro_with_dividend(
        S=S,
        K=K,
        T=T,
        r=r,
        sgma=sgma / np.sqrt(3),
        dividend_yield=0.5 * (r + dividend_yield + sgma**2 / 6),
    )


def geometric_asian_average_price_p(S, K, T, r, sgma, dividend_yield):
    return p_euro_with_dividend(
        S=S,
        K=K,
        T=T,
        r=r,
        sgma=sgma / np.sqrt(3),
        dividend_yield=0.5 * (r + dividend_yield + sgma**2 / 6),
    )


def geometric_asian_average_strike_c(S, K, T, r, sgma, dividend_yield):
    return c_euro_with_dividend(
        S=S,
        K=K,
        T=T,
        dividend_yield=dividend_yield,
        sgma=sgma * np.sqrt(T / 3),
        r=0.5 * (r + dividend_yield + sgma**2 / 6),
    )


def geometric_asian_average_strike_p(S, K, T, r, sgma, dividend_yield):
    return p_euro_with_dividend(
        S=S,
        K=K,
        T=T,
        dividend_yield=dividend_yield,
        sgma=sgma * np.sqrt(T / 3),
        r=0.5 * (r + dividend_yield + sgma**2 / 6),
    )

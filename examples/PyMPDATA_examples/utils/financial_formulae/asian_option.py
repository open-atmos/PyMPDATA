# pylint: disable=line-too-long
"""
Closed-forms for geometric Asian options are taken from:
[McDonald 2013, "Derivatives Markets", Appendix 19A](https://media.pearsoncmg.com/ph/bp/bridgepages/teamsite/mcdonald/McDonald-web-19-A.pdf)
"""

import numpy as np
from scipy.stats import norm

from .Black_Scholes_1973 import c_euro_with_dividend, p_euro_with_dividend

# def geometric_asian_average_price_c(S, K, T, r, sgma, dividend_yield):
#     return c_euro_with_dividend(
#         S=S,
#         K=K,
#         T=T,
#         r=r,
#         sgma=sgma / np.sqrt(3),
#         dividend_yield=0.5 * (r + dividend_yield + sgma**2 / 6),
#     )
#
#
# def geometric_asian_average_price_p(S, K, T, r, sgma, dividend_yield):
#     return p_euro_with_dividend(
#         S=S,
#         K=K,
#         T=T,
#         r=r,
#         sgma=sgma / np.sqrt(3),
#         dividend_yield=0.5 * (r + dividend_yield + sgma**2 / 6),
#     )


# def geometric_asian_average_strike_c(S, K, T, r, sgma, dividend_yield):
#     return c_euro_with_dividend(
#         S=S,
#         K=K,
#         T=T,
#         dividend_yield=dividend_yield,
#         sgma=sgma * np.sqrt(T / 3),
#         r=0.5 * (r + dividend_yield + sgma**2 / 6),
#     )
#
#
# def geometric_asian_average_strike_p(S, K, T, r, sgma, dividend_yield):
#     return p_euro_with_dividend(
#         S=S,
#         K=K,
#         T=T,
#         dividend_yield=dividend_yield,
#         sgma=sgma * np.sqrt(T / 3),
#         r=0.5 * (r + dividend_yield + sgma**2 / 6),
#     )


def calculate_d_asian(S, K, T, r, sgma):
    d_1 = (np.log(S / K) + 0.5 * (r + (sgma**2) / 6) * T) / (sgma * np.sqrt(T / 3))
    d_2 = d_1 - sgma * np.sqrt(T / 3)
    return d_1, d_2


def geometric_asian_average_price_c(S, K, T, r, sgma, dividend_yield):
    d_1, d_2 = calculate_d_asian(S, K, T, r, sgma)
    call_value = S * np.exp(-0.5 * (r + (sgma**2) / 6) * T) * norm.cdf(
        d_1
    ) - K * np.exp(-r * T) * norm.cdf(d_2)
    return call_value


def geometric_asian_average_price_p(S, K, T, r, sgma, dividend_yield):
    d_1, d_2 = calculate_d_asian(S, K, T, r, sgma)
    put_value = K * np.exp(-r * T) * norm.cdf(-d_2) - S * np.exp(
        -0.5 * (r + (sgma**2) / 6) * T
    ) * norm.cdf(-d_1)
    return put_value

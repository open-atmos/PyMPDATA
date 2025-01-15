import numpy as np
from scipy.special import erf  # pylint: disable=no-name-in-module


def N(x: float):
    return (1 + erf(x / np.sqrt(2))) / 2


def c_euro(S: np.ndarray, K: float, T: float, r: float, b: float, sgma: float):
    d1 = (np.log(S / K) + (b + sgma * sgma / 2) * T) / sgma / np.sqrt(T)
    d2 = d1 - sgma * np.sqrt(T)
    return S * np.exp(b - r) * N(d1) - K * np.exp(-r * T) * N(d2)


def p_euro(S: np.ndarray, K: float, T: float, r: float, b: float, sgma: float):
    d1 = (np.log(S / K) + (b + sgma * sgma / 2) * T) / sgma / np.sqrt(T)
    d2 = d1 - sgma * np.sqrt(T)
    return K * np.exp(-r * T) * N(-d2) - S * np.exp((b - r) * T) * N(-d1)

def c_euro_with_dividend(S: np.ndarray, K: float, T: float, r: float, sgma: float, dividend_yield: float):
    b = r - dividend_yield
    return c_euro(S, K, T, r, b, sgma)

def p_euro_with_dividend(S: np.ndarray, K: float, T: float, r: float, sgma: float, dividend_yield: float):
    b = r - dividend_yield
    return p_euro(S, K, T, r, b, sgma)

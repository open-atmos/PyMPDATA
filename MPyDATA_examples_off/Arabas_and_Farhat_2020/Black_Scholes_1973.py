import numpy as np
import scipy.special as sp


def _N(x: float):
    return (1 + sp.erf(x / np.sqrt(2))) / 2


def c_euro(S: np.ndarray, K: float, T: float, r: float, b: float, sgma: float):
    d1 = (np.log(S / K) + (b + sgma * sgma / 2) * T) / sgma / np.sqrt(T)
    d2 = d1 - sgma * np.sqrt(T)
    return S * np.exp(b - r) * _N(d1) - K * np.exp(-r * T) * _N(d2)


def p_euro(S: np.ndarray, K: float, T: float, r: float, b: float, sgma: float):
    d1 = (np.log(S / K) + (b + sgma * sgma / 2) * T) / sgma / np.sqrt(T)
    d2 = d1 - sgma * np.sqrt(T)
    return K * np.exp(-r * T) * _N(-d2) - S * np.exp((b - r) * T) * _N(-d1)

import PyMPDATA_examples.Arabas_and_Farhat_2020.Black_Scholes_1973 as BS
import numpy as np


def _phi(S: [np.ndarray, float], gamma: float, H: float, I: float, r: float, b: float, var: float, T: float):
    lmbd = (-r + gamma * b + .5 * gamma * (gamma - 1) * var) * T
    d = -(np.log(S / H) + (b + (gamma - .5) * var) * T) / np.sqrt(var * T)
    kappa = 2 * b / var + (2 * gamma - 1)
    return np.exp(lmbd) * np.power(S, gamma) *(BS._N(d) - pow((I / S), kappa) * BS._N(d - 2 * np.log(I / S) / np.sqrt(var * T)))


def c_amer(S: [np.ndarray, float], K: [float, np.ndarray], T: float, r: float, b: float, sgma: float):
    if b >= r:
        return BS.c_euro(S, K=K, T=T, r=r, b=b, sgma=sgma)

    var = sgma*sgma
    beta = (.5 - b/var) + np.sqrt(pow((b/var - .5), 2) + 2 * r/var)
    BInf = beta / (beta - 1) * K
    B0 = np.maximum(K, r / (r - b) * K)
    ht = -(b*T + 2 * sgma * np.sqrt(T)) * B0 / (BInf - B0)
    I = B0 + (BInf - B0) * (1 - np.exp(ht))
    alpha = (I-K) * pow(I, -beta)

    return np.where(
        S >= I,
        S - K,
        alpha * np.power(S, beta) + (
            - alpha * _phi(S, gamma=beta, H=I, I=I, r=r, b=b, var=var, T=T)
            +         _phi(S, gamma=1,    H=I, I=I, r=r, b=b, var=var, T=T)
            -         _phi(S, gamma=1,    H=K, I=I, r=r, b=b, var=var, T=T)
            -     K * _phi(S, gamma=0,    H=I, I=I, r=r, b=b, var=var, T=T)
            +     K * _phi(S, gamma=0,    H=K, I=I, r=r, b=b, var=var, T=T)
        )
    )


def p_amer(S: [np.ndarray, float], K: float, T: float, r: float, b: float, sgma: float):
    return c_amer(K, K=S, T=T, r=r - b, b=-b, sgma=sgma)

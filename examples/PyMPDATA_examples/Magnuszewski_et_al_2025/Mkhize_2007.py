import numpy as np
from scipy.stats import norm

def geometric_mkhize(s_t=100, K=100, r=0.008, sigma=0.2, T=30, T_0=0):
    d_1 = (np.log(s_t / K) + 0.5 * (r + (sigma ** 2) / 6) * (T - T_0)) / (sigma * np.sqrt((T - T_0) / 3))
    d_2 = d_1 - sigma * np.sqrt((T - T_0) / 3)
    C_0 = s_t * np.exp(-0.5 * (r + (sigma ** 2) / 6) * (T - T_0)) * norm.cdf(d_1) - K * np.exp(
        -r * (T - T_0)) * norm.cdf(d_2)
    P_0 = K * np.exp(-r * (T - T_0)) * norm.cdf(-d_2) - s_t * np.exp(
        -0.5 * (r + (sigma ** 2) / 6) * (T - T_0)) * norm.cdf(-d_1)
    return C_0
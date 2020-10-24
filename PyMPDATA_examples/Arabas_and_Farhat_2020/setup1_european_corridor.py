import PyMPDATA_examples.Arabas_and_Farhat_2020.Black_Scholes_1973 as BS73
import numpy as np


class Setup:
    S0 = 55
    T = 0.5
    amer = False
    S_min = 10
    S_max = 2000
    sigma = 0.6
    r = 0.008
    K1 = 75
    K2 = 175
    S_match = 175

    def __init__(self, *, n_iters=2, l2_opt=2, C_opt=0.034):
        self.n_iters = n_iters
        self.l2_opt = l2_opt
        self.C_opt = C_opt

    def payoff(self, S: np.ndarray):
        return np.exp(-self.r * self.T) * (np.maximum(0, self.K2 - S) - np.maximum(0, self.K1 - S))

    def analytical_solution(self, S: np.ndarray):
        return (
                BS73.p_euro(S, K=self.K2, T=self.T, r=self.r, b=self.r, sgma=self.sigma) -
                BS73.p_euro(S, K=self.K1, T=self.T, r=self.r, b=self.r, sgma=self.sigma)
        )

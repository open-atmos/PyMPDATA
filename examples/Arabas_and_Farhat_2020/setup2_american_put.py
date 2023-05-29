import numpy as np
from pystrict import strict
import PyMPDATA_examples.Arabas_and_Farhat_2020.Black_Scholes_1973 as BS73
import PyMPDATA_examples.Arabas_and_Farhat_2020.Bjerksund_and_Stensland_1993 as BS93


@strict
class Settings:
    amer = True
    l2_opt = 2.05
    S_min = .05
    S_max = 500
    K = 100
    r = .08
    sigma = .2
    n_iters = 2

    def __init__(self, *, T: float, C_opt: float, S0: float):
        self.T = T
        self.C_opt = C_opt
        self.S0 = S0
        self.S_match = S0

    def payoff(self, S: np.ndarray):
        return np.exp(-self.r * self.T) * (np.maximum(0, self.K - S))

    def analytical_solution(self, S: [np.ndarray, float], amer=True):
        if not amer:
            return BS73.p_euro(S, K=self.K, T=self.T, r=self.r, b=self.r, sgma=self.sigma)
        return BS93.p_amer(S, K=self.K, T=self.T, r=self.r, b=self.r, sgma=self.sigma)

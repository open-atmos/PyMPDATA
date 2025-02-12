import numpy as np
import PyMPDATA_examples.utils.financial_formulae.Black_Scholes_1973 as BS73
from pystrict import strict


@strict
class Settings:
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

    def __init__(self, *, n_iters: int = 2, l2_opt: int = 2, C_opt: float = 0.034):
        self.n_iters = n_iters
        self.l2_opt = l2_opt
        self.C_opt = C_opt

    def terminal_condition(self, S: np.ndarray):
        return np.exp(-self.r * self.T) * self.payoff(S)

    def payoff(self, S: np.ndarray):
        return np.maximum(0, self.K2 - S) - np.maximum(0, self.K1 - S)

    def analytical_solution(self, S: np.ndarray):
        return BS73.p_euro(
            S, K=self.K2, T=self.T, r=self.r, b=self.r, sgma=self.sigma
        ) - BS73.p_euro(S, K=self.K1, T=self.T, r=self.r, b=self.r, sgma=self.sigma)

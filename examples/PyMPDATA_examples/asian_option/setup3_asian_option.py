import numpy as np
import PyMPDATA_examples.asian_option.Mkhize_2007 as MKH
from pystrict import strict
from scipy.stats import norm


@strict
class Settings:
    # S0 = 55
    T = 0.5
    # amer = False
    S_min = 10
    S_max = 2000
    sigma = 0.6
    r = 0.008
    K1 = 100
    # K2 = 175
    S_match = 175

    def __init__(self, *, n_iters: int = 2, l2_opt: int = 2, C_opt: float = 0.034):
        self.n_iters = n_iters
        self.l2_opt = l2_opt
        self.C_opt = C_opt

    def payoff(self, A: np.ndarray):
        return np.maximum(0, A - self.K1)

    def terminal_value(self, A: np.ndarray):
        return np.exp(-self.r * self.T) * self.payoff(A)

    def analytical_solution(self, S: np.ndarray):
        return MKH.geometric_mkhize(
            s_t=S, K=self.K1, r=self.r, sigma=self.sigma, T=self.T, T_0=0
        )

        # return BS73.p_euro(
        #     S, K=self.K2, T=self.T, r=self.r, b=self.r, sgma=self.sigma
        # ) - BS73.p_euro(S, K=self.K1, T=self.T, r=self.r, b=self.r, sgma=self.sigma)

import numpy as np
import PyMPDATA_examples.asian_option.Mkhize_2007 as MKH
from pystrict import strict
from scipy.stats import norm


@strict
class Settings:
    # S0 = 55
    T = 1
    # amer = False
    S_min = 1
    S_max = 1600
    sigma = 0.3
    r = 0.08
    K1 = 40
    # K2 = 175
    S_match = 40

    def __init__(self, *, n_iters: int = 2, l2_opt: int = 2, C_opt: float = 0.034):
        self.n_iters = n_iters
        self.l2_opt = l2_opt
        self.C_opt = C_opt

    def payoff(self, A: np.ndarray):
        # A is a 1d array,
        # the payoff is the max of the average price minus the strike price
        # however, we need to transform it into a 2d array, that is square
        print(f"{np.max(A)=}, {np.min(A)=}")
        payoff = np.maximum(0, A - self.K1)
        return np.repeat([payoff], payoff.shape[0], axis=0)
        # payoff_matrix = np.zeros((payoff.shape[0], payoff.shape[0]))
        # for i in range(payoff.shape[0]):
        #     payoff_matrix[i, :] = payoff
        # return payoff_matrix
        # return np.maximum(0, A - self.K1)

    def terminal_value(self, A: np.ndarray):
        return np.exp(-self.r * self.T) * self.payoff(A)

    # def terminal_value(self, A: np.ndarray):
    #     # put zeros everrywhere except for a 2x2 square in the middle where we put 1
    #     cond = np.zeros_like(A, dtype=np.float64)
    #     print(f"{cond.shape=}")
    #     # print(f"{A.shape[0]//2-1}, {A.shape[0]//2+1}")
    #     cond[
    #         A.shape[0] // 3 - 1 : A.shape[0] // 3 + 1,
    #         A.shape[1] // 2 - 1 : A.shape[1] // 2 + 1,
    #     ] = 1
    #     return cond

    def analytical_solution(self, S: np.ndarray):
        return MKH.geometric_mkhize(
            s_t=S, K=self.K1, r=self.r, sigma=self.sigma, T=self.T, T_0=0
        )

        # return BS73.p_euro(
        #     S, K=self.K2, T=self.T, r=self.r, b=self.r, sgma=self.sigma
        # ) - BS73.p_euro(S, K=self.K1, T=self.T, r=self.r, b=self.r, sgma=self.sigma)

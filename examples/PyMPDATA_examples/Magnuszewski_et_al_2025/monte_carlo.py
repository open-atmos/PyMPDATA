from abc import abstractmethod
from typing import Callable

import numpy as np
from tqdm import tqdm


class BSModel:
    def __init__(self, S0, r, sigma, T, M):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.sigma2 = sigma * sigma
        self.b = r - 0.5 * self.sigma2
        self.T = T
        self.M = M
        self.t = np.linspace(0, T, M)
        self.bt = self.b * self.t
        self.sqrt_tm = np.sqrt(T / M)

    def generate_path(self):
        W = np.random.standard_normal(self.M)
        W = np.cumsum(W) * self.sqrt_tm
        S = self.S0 * np.exp(self.bt + self.sigma * W)
        return S


class PathDependentOption:
    def __init__(self, T):
        self.T = T
        self.payoff: Callable[[np.ndarray], float] = lambda path: 0.0

    def price_by_mc(self, model: BSModel, N):
        sum_ct = 0.0
        for _ in range(N):
            path = model.generate_path()
            sum_ct += self.payoff(path)
        return np.exp(-model.r * self.T) * (sum_ct / N)


class FixedStrikeArithmeticAsianOption(PathDependentOption):
    def __init__(self, T, K, type="call"):
        super().__init__(T)
        self.K = K

        if type == "call":
            self.payoff = lambda path: max(np.mean(path) - K, 0)
        elif type == "put":
            self.payoff = lambda path: max(K - np.mean(path), 0)
        else:
            raise ValueError("Invalid option type")


class FixedStrikeGeometricAsianOption(PathDependentOption):
    def __init__(self, T, K, type="call"):
        super().__init__(T)
        self.K = K

        if type == "call":
            self.payoff = lambda path: max(np.exp(np.mean(np.log(path))) - K, 0)
        elif type == "put":
            self.payoff = lambda path: max(K - np.exp(np.mean(np.log(path))), 0)
        else:
            raise ValueError("Invalid option type")

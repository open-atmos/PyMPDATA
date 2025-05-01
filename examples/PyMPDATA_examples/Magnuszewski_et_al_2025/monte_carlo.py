"""
This code is a Python numba-fied implementation of the Monte Carlo method
for pricing Asian options taken from
[Numerical Methods in Finance with C++](https://doi.org/10.1017/CBO9781139017404)
"""

from functools import cached_property, lru_cache, partial
from typing import Callable

import numba
import numpy as np

jit = partial(numba.jit, fastmath=True, error_model="numpy", cache=True, nogil=True)

# pylint: disable=too-few-public-methods


class BSModel:
    def __init__(self, S0, r, sigma, T, M, seed):
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
        self.seed = seed

    @cached_property
    def generate_path(self):
        M = self.M
        S0 = self.S0
        bt = self.bt
        sigma = self.sigma
        sqrt_tm = self.sqrt_tm
        seed = self.seed

        @jit
        def numba_seed():
            np.random.seed(seed)

        if seed is not None:
            numba_seed()

        @jit
        def body(path):
            path[:] = S0 * np.exp(
                bt + sigma * np.cumsum(np.random.standard_normal(M)) * sqrt_tm
            )

        return body


class PathDependentOption:
    def __init__(self, T, model, N):
        self.T = T
        self.model = model
        self.N = N
        self.payoff: Callable[[np.ndarray], float] = lambda path: 0.0

    @cached_property
    def price_by_mc(self):
        T = self.T
        model_generate_path = self.model.generate_path
        model_r = self.model.r
        payoff = self.payoff
        M = self.model.M
        N = self.N

        @jit
        def body():
            sum_ct = 0.0
            path = np.empty(M)
            for _ in range(N):
                model_generate_path(path)
                sum_ct += payoff(path)
            return np.exp(-model_r * T) * (sum_ct / N)

        return body


@lru_cache
def make_payoff(K: float, option_type: str, average_type: str = "arithmetic"):
    assert average_type in ["arithmetic", "geometric"]
    if average_type != "arithmetic":
        raise NotImplementedError("Only arithmetic average is supported")
    if option_type == "call":

        @jit
        def payoff(path):
            return max(np.mean(path) - K, 0)

    elif option_type == "put":

        @jit
        def payoff(path):
            return max(K - np.mean(path), 0)

    else:
        raise ValueError("Invalid option")
    return payoff


class FixedStrikeArithmeticAsianOption(PathDependentOption):
    def __init__(self, T, K, variant, model, N):
        super().__init__(T, model, N)
        self.K = K
        self.payoff = make_payoff(K, variant)


class FixedStrikeGeometricAsianOption(PathDependentOption):
    def __init__(self, T, K, variant, model, N):
        super().__init__(T, model, N)
        self.K = K

        if variant == "call":
            self.payoff = lambda path: max(np.exp(np.mean(np.log(path))) - K, 0)
        elif variant == "put":
            self.payoff = lambda path: max(K - np.exp(np.mean(np.log(path))), 0)
        else:
            raise ValueError("Invalid option type")

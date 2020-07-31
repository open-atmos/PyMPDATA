from MPyDATA_examples.Arabas_and_Farhat_2020.options import OPTIONS
from MPyDATA import Factories
from MPyDATA import ExtrapolatedBoundaryCondition
from MPyDATA import Options
import numpy as np


class Simulation:
    def __init__(self, setup):
        self.setup = setup

        sigma2 = pow(setup.sigma, 2)
        dx_opt = abs(setup.C_opt / (.5 * sigma2 - setup.r) * setup.l2_opt * sigma2)
        dt_opt = pow(dx_opt, 2) / sigma2 / setup.l2_opt
    
        # adjusting dt so that nt is integer
        self.dt = setup.T
        self.nt = 0
        while self.dt > dt_opt:
            self.nt += 1
            self.dt = setup.T / self.nt
    
        # adjusting dx to match requested l^2
        dx = np.sqrt(setup.l2_opt * self.dt) * setup.sigma

        # calculating actual u number and lambda
        self.C = - (.5 * sigma2 - setup.r) * (-self.dt) / dx
        self.l2 = dx * dx / sigma2 / self.dt
    
        # adjusting nx and setting S_beg, S_end
        S_beg = setup.S_match
        self.nx = 1
        while S_beg > setup.S_min:
            self.nx += 1
            S_beg = np.exp(np.log(setup.S_match) - self.nx * dx)

        self.ix_match = self.nx
    
        S_end = setup.S_match
        while S_end < setup.S_max:
            self.nx += 1
            S_end = np.exp(np.log(S_beg) + (self.nx-1) * dx)

        # asset price
        self.S = np.exp(np.log(S_beg) + np.arange(self.nx) * dx)

        self.mu_coeff = (0.5 / self.l2,)
        self.solvers = {}
        self.solvers[1] = Factories.advection_diffusion_1d(
            advectee=setup.payoff(self.S),
            advector=self.C,
            options=Options(n_iters=1, non_zero_mu_coeff=True),
            boundary_conditions=(ExtrapolatedBoundaryCondition(),)
        )
        self.solvers[2] = Factories.advection_diffusion_1d(
            advectee=setup.payoff(self.S),
            advector=self.C,
            options=Options(**OPTIONS),
            boundary_conditions=(ExtrapolatedBoundaryCondition(),)
        )

    def run(self, n_iters: int):
        if self.setup.amer:
            psi = self.solvers[n_iters].advectee.get()
            f_T = np.empty_like(psi)
            f_T[:] = psi[:] / np.exp(-self.setup.r * self.setup.T)
            t = self.setup.T
            for _ in range(self.nt):
                self.solvers[n_iters].advance(1, self.mu_coeff)
                psi = self.solvers[n_iters].advectee.get()
                t -= self.dt
                psi[:] += np.maximum(psi[:], f_T[:]/np.exp(self.setup.r * t)) - psi[:]
        else:
            self.solvers[n_iters].advance(self.nt, self.mu_coeff)

        return self.solvers[n_iters].advectee.get()

    def terminal_value(self):
        return self.solvers[1].advectee.get()

from MPyDATA_examples.Arabas_and_Farhat_2019.options import OPTIONS
from MPyDATA.mpdata_factory import MPDATAFactory
# TODO
#from MPyDATA.arakawa_c.boundary_conditions.extrapolated import ExtrapolatedLeft, ExtrapolatedRight
import numpy as np
from MPyDATA.options import Options

class Simulation:
    def __init__(self, setup):
        self.setup = setup

        sgma2 = pow(setup.sgma, 2)
        dx_opt = abs(setup.C_opt / (.5 * sgma2 - setup.r) * setup.l2_opt * sgma2)
        dt_opt = pow(dx_opt, 2) / sgma2 / setup.l2_opt
    
        # adjusting dt so that nt is integer
        self.dt = setup.T
        self.nt = 0
        while self.dt > dt_opt:
            self.nt += 1
            self.dt = setup.T / self.nt
    
        # adjusting dx to match requested l^2
        dx = np.sqrt(setup.l2_opt * self.dt) * setup.sgma

        # calculating actual u number and lambda
        self.C = - (.5 * sgma2 - setup.r) * (-self.dt) / dx
        self.l2 = dx * dx / sgma2 / self.dt
    
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

        mu_coeff = 0.5 / self.l2

        self.solvers = {}
        self.solvers[1] = MPDATAFactory.advection_diffusion_1d(
            psi=setup.payoff(self.S),
            C=self.C,
            options=OPTIONS,
#            boundary_conditions=((ExtrapolatedLeft(), ExtrapolatedRight()),)
            mu_coeff=mu_coeff
        )
        self.solvers[2] = MPDATAFactory.advection_diffusion_1d(
            psi=setup.payoff(self.S),
            C=self.C,
            options=Options(n_iters=1),
#            boundary_conditions=((ExtrapolatedLeft(), ExtrapolatedRight()),)
            mu_coeff=mu_coeff
        )


    def run(self, n_iters: int):
        psi = self.solvers[n_iters].curr.get()
        f_T = np.empty_like(psi)
        f_T[:] = psi[:] / np.exp(-self.setup.r * self.setup.T)
        t = self.setup.T

        for _ in range(self.nt):
            self.solvers[n_iters].step(1)
            if self.setup.amer:
                psi = self.solvers[n_iters].curr.get()
                t -= self.dt
                psi[:] += np.maximum(psi[:], f_T[:]/np.exp(self.setup.r * t)) - psi[:]

        return self.solvers[n_iters].curr.get()

    def terminal_value(self):
        return self.solvers[1].curr.get()


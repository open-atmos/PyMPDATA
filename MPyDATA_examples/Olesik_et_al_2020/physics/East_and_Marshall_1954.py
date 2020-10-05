import numpy as np
from scipy import special


class SizeDistribution:
    def __init__(self, si):
        self.si = si
        self.n0 = 700 / si.centimetre**3
        self.kappa = 22

    def pdf(self, r):
        return (
                (self.n0 * self.si.micrometre) / r *
                np.exp(-self.kappa * (np.log10(r / (7 * self.si.micrometre)) ** 2)) *
                (1 / self.si.micrometre))





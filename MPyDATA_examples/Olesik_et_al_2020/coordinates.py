"""
Created at 22.07.2019

@author: Michael Olesik
@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np


class x_id:
    def x(self, r):
        return r

    def r(self, x):
        return x

    def dx_dr(self, r):
        return r**0


class x_ln:
    def __init__(self):
        self.r0 = 1

    def x(self, r):
        return np.log(r / self.r0)

    def r(self, x):
        return self.r0 * np.exp(x)

    def dx_dr(self, r):
        return 1/r


class x_p2:
    def x(self, r):
        return r**2

    def r(self, x):
        return np.sqrt(np.where(x < 0, 1e10, x))

    def dx_dr(self, r):
        return 2*r

class x_p3:
    def x(self,r):
        return r**3

    def r(self, x):
        return np.power(x, 1/3)

    def dx_dr(self, r):
        return 3*r**2


class x_log_of_p3:
    def __init__(self, base = np.e):
        self.r0 = 1
        self.base = base

    def x(self, r):
        return np.log(r**3 / self.r0**3) / np.log(self.base)

    def r(self, x):
        return self.r0 * self.base**(x/3)

    def dx_dr(self, r):
        return 3 / r / np.log(self.base)







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


class x_log_of_pn:
    def __init__(self, base = np.e, n = 3):
        self.r0 = 1
        self.base = base
        self.n = n

    def x(self, r):
        return np.log(r**self.n / self.r0**self.n) / np.log(self.base)

    def r(self, x):
        return self.r0 * self.base**(x/self.n)

    def dx_dr(self, r):
        return self.n / r / np.log(self.base)







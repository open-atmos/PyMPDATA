"""
Created at 22.07.2019

@author: Michael Olesik
@author: Piotr Bartman
@author: Sylwester Arabas
"""

import numpy as np

# TODO: add moment function
class x_id:
    def x(self, r):
        return r

    def r(self, x):
        return x

    def dx_dr(self, r):
        return r**0

    def moment_of_r_integral(self, x, k):
        return 1 / (k + 1) * x ** (k + 1)




class x_p2:
    def x(self, r):
        return r**2

    def r(self, x):
        return np.sqrt(np.where(x < 0, 1e10, x))

    def dx_dr(self, r):
        return 2*r

    def moment_of_r_integral(self, x, k):
        return 2 / (k + 2) * x**((k+2)/2)

class x_p3:
    def x(self,r):
        return r**3

    def r(self, x):
        return np.power(x, 1/3)

    def dx_dr(self, r):
        return 3*r**2

    def moment_of_r_integral(self, x, k):
        return 3 / (k + 3) * x**((k+3)/3)


class x_log_of_pn:
    def __init__(self, r0, base = np.e, n = 3):
        self.r0 = r0
        self.base = base
        self.n = n

    def x(self, r):
        return np.log(r**self.n / self.r0**self.n) / np.log(self.base)

    def r(self, x):
        return self.r0 * self.base**(x/self.n)

    def dx_dr(self, r):
        return self.n / r / np.log(self.base)

    def moment_of_r_integral(self, x, k):
        return self.r0**k * self.n / (k * np.log(self.base)) * self.base**(k / self.n * x)







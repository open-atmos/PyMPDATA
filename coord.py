import numpy


class x_id:
    def __init__(self, si):
        self.si = si

    def x(self, r):
        return r

    def r(self, x):
        return x

    def dx_dr(self, r):
        return r**0 * self.si.dimensionless


class x_ln:
    def __init__(self, si):
        self.si = si
        self.r0 = 1 * si.metre
        self.x_unit = si.dimensionless

    def x(self, r):
        return numpy.log(r / self.r0)

    def r(self, x):
        return self.r0 * numpy.exp(x)

    def dx_dr(self, r):
        return 1/r


class x_p2:
    def __init__(self, _):
        pass

    def x(self, r):
        return r**2

    def r(self, x):
        return numpy.sqrt(x)

    def dx_dr(self, r):
        return 2*r





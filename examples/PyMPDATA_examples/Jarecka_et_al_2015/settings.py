from pystrict import strict

from PyMPDATA import Options


@strict
class Settings:
    def __init__(self):
        self.dt = 0.01
        self.dx = 0.05
        self.dy = 0.05
        self.nx = 401
        self.ny = 401
        self.eps = 1e-7
        self.lx0 = 2
        self.ly0 = 1
        self.options = Options(nonoscillatory=True, infinite_gauge=True)

    @property
    def nt(self):
        return int(7 / self.dt)

    @property
    def outfreq(self):
        return int(1 / self.dt)

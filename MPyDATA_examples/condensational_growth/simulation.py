from MPyDATA.mpdata_factory import MPDATAFactory


class Simulation:
    @staticmethod
    def __mgn(quantity, unit):
        return quantity.to(unit).magnitude

    def __init__(self, coord, setup, opts):
        # units of calculation
        self.__t_unit = setup.si.seconds
        self.__r_unit = setup.si.metres
        self.__n_unit = setup.si.metres ** -3 / self.__r_unit

        self.solver, self.__r, self.__rh, = MPDATAFactory.equilibrium_growth_C_1d(
            setup.nr,
            self.__mgn(setup.r_min, self.__r_unit),
            self.__mgn(setup.r_max, self.__r_unit),
            self.__mgn(setup.dt, self.__t_unit),
            coord,
            lambda r: self.__mgn(setup.cdf(r * self.__r_unit), self.__n_unit * self.__r_unit),
            lambda r: self.__mgn(setup.drdt(r * self.__r_unit), self.__r_unit / self.__t_unit),
            opts
        )

    def step(self, n_iters: int, debug=False):
        self.solver.step(n_iters=n_iters, debug=debug)

    @property
    def r(self):
        return self.__r * self.__r_unit

    @property
    def rh(self):
        return self.__rh * self.__r_unit

    @property
    def n(self):
        return self.solver.arrays.curr.get() * self.__n_unit

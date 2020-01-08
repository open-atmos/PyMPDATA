from MPyDATA.mpdata_factory import MPDATAFactory


class Simulation:
    @staticmethod
    def __mgn(quantity, unit):
        return quantity.to(unit).magnitude

    def __init__(self, coord, setup, opts):
        self.t_unit = setup.si.seconds
        self.r_unit = setup.si.metres
        self.n_unit = setup.si.metres**-3
        self.solver, self.r = MPDATAFactory.TODO(
            setup.nr,
            self.__mgn(setup.r_min, self.r_unit),
            self.__mgn(setup.r_max, self.r_unit),
            self.__mgn(setup.dt, self.t_unit),
            coord,
            lambda r: self.__mgn(setup.cdf(r * self.r_unit), self.n_unit),
            lambda r: self.__mgn(setup.drdt(r * self.r_unit), self.r_unit / self.t_unit),
            opts
        )

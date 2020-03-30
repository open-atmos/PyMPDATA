from MPyDATA.mpdata_factory import MPDATAFactory



class Simulation:
    @staticmethod
    def __mgn(quantity, unit):
        return quantity.to(unit).magnitude

    def __init__(self, setup, grid_layout, psi_coord, opts):
        self.setup = setup
        self.psi_coord = psi_coord

        # units of calculation
        self.__t_unit = self.setup.si.seconds
        self.__r_unit = self.setup.si.micrometre
        self.__n_unit = self.setup.si.centimetres**-3 / self.setup.si.micrometre

        self.solver, self.__r, self.__rh, self.dx = MPDATAFactory.equilibrium_growth_C_1d(
            self.setup.nr,
            self.__mgn(self.setup.r_min, self.__r_unit),
            self.__mgn(self.setup.r_max, self.__r_unit),
            self.__mgn(self.setup.dt, self.__t_unit),
            grid_layout,
            psi_coord,
            lambda r: self.__mgn(self.setup.pdf(r * self.__r_unit), self.__n_unit),
            lambda r: self.__mgn(self.setup.drdt(r * self.__r_unit), self.__r_unit / self.__t_unit),
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
        psi = self.solver.arrays.curr.get()
        n = psi * self.psi_coord.dx_dr(self.__r)
        return n * self.__n_unit


from MPyDATA import Factories
from functools import lru_cache
from scipy import optimize
from MPyDATA_examples.Olesik_et_al_2020.physics import equilibrium_drop_growth


class Simulation:
    @staticmethod
    def __mgn(quantity, unit):
        return quantity.to(unit).magnitude

    def __init__(self, setup, grid_layout, psi_coord, opts, GC_max):
        self.setup = setup
        self.psi_coord = psi_coord

        # units of calculation
        self.__t_unit = self.setup.si.seconds
        self.__r_unit = self.setup.si.micrometre
        self.__n_unit = self.setup.si.centimetres ** -3 / self.setup.si.micrometre

        self.solver, self.__r, self.__rh, self.dx, dt = Factories.condensational_growth(
            self.setup.nr,
            self.__mgn(self.setup.r_min, self.__r_unit),
            self.__mgn(self.setup.r_max, self.__r_unit),
            GC_max,
            grid_layout,
            psi_coord,
            lambda r: self.__mgn(self.setup.pdf(r * self.__r_unit), self.__n_unit),
            lambda r: self.__mgn(self.setup.drdt(r * self.__r_unit), self.__r_unit / self.__t_unit),
            opts
        )

        self.out_steps = Simulation.find_out_steps(setup=self.setup, dt=dt)
        self.dt = dt * self.__t_unit

    def step(self, nt):
        self.solver.advance(nt)

    @property
    def r(self):
        return self.__r * self.__r_unit

    @property
    def rh(self):
        return self.__rh * self.__r_unit

    @property
    def n(self):
        psi = self.solver.curr.get()
        n = psi * self.psi_coord.dx_dr(self.__r)
        return n * self.__n_unit

    @staticmethod
    def find_out_steps(setup, dt):
        out_steps = []
        for mr in setup.mixing_ratios:
            @lru_cache()
            def findroot(ti):
                return (mr - setup.mixing_ratio(
                    equilibrium_drop_growth.PdfEvolver(setup.pdf, setup.drdt, ti * t_unit))).magnitude

            t_unit = setup.si.second
            t = optimize.brentq(findroot, 0, (1 * setup.si.hour).to(t_unit).magnitude)
            out_steps.append(int((t / dt)))
        return out_steps



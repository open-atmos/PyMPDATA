from MPyDATA.mpdata_factory import MPDATAFactory
from MPyDATA_examples.Molenkamp_test_as_in_Jaruga_et_al_2015_Fig_12.setup import Setup
import numpy as np


def from_pdf_2d(pdf: callable, xrange: list, yrange: list, gridsize: list):
    z = np.empty(gridsize)
    dx, dy = (xrange[1] - xrange[0]) / gridsize[0], (yrange[1] - yrange[0]) / gridsize[1]
    for i in range(gridsize[0]):
        for j in range(gridsize[1]):
            z[i, j] = pdf(
                xrange[0] + dx * (i + .5),
                yrange[0] + dy * (j + .5)
            )

    x = np.linspace(xrange[0] + dx / 2, xrange[1] - dx / 2, gridsize[0])
    y = np.linspace(yrange[0] + dy / 2, yrange[1] - dy / 2, gridsize[1])
    return x, y, z


class Simulation:
    def __init__(self, setup: Setup):

        x, y, z = from_pdf_2d(setup.pdf, xrange=setup.xrange, yrange=setup.yrange, gridsize=setup.grid)
        self.mpdata = MPDATAFactory.constant_2d(data=z, C=(-.5, .25))
        self.nt = setup.nt

    @property
    def state(self):
        return self.mpdata.arrays.curr.get().copy()

    def run(self):
        self.mpdata.step(self.nt)

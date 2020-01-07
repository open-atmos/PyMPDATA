from MPyDATA.mpdata_factory import MPDATAFactory
from MPyDATA_examples.condensational_growth import coord
import numpy as np


class Simulation:
    @staticmethod
    def __mgn(quantity):
        return quantity.to_base_units().magnitude

    def __init__(self, coord, setup, opts):
      solver = MPDATAFactory.TODO(
          setup.nr,
          self.__mgn(setup.r_min),
          self.__mgn(setup.r_max),
          self.__mgn(setup.dt),
          coord,
          setup.cdf,
          setup.drdt,
          opts
      )